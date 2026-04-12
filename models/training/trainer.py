import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    EarlyStoppingCallback,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from crisp_smiles.main import CRISPSmiles

CRISPSmilesConverter = CRISPSmiles()


class SmilesPreprocessing(TypedDict):
    input_smiles_syntax: str
    input_smiles_type: str
    crisp_input_deferred: bool


def randomize_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Generate a randomized (non-canonical) SMILES string for a molecule.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        Optional[str]: A randomized SMILES string, or None on failure.
    """
    try:
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return None


def process_single_smiles(
    smiles,
    smiles_preprocessing: SmilesPreprocessing,
    max_atoms: int = 150,
) -> Optional[str]:
    """
    Process a single molecule into an (input, output) SMILES pair.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        mode (str): One of MODE_RANDOM_TO_CANONICAL or MODE_RANDOM_TO_RANDOM.
        max_atoms (int): Skip molecules with more atoms than this.

    Returns:
        Optional[Tuple[str, str]]: (input_smiles, output_smiles) or None if skipped.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return None

    if not smiles_preprocessing:
        return None

    input_smiles_syntax = smiles_preprocessing["input_smiles_syntax"]
    input_smiles_type = smiles_preprocessing["input_smiles_type"]
    crisp_input_deferred = smiles_preprocessing["crisp_input_deferred"]

    if input_smiles_syntax == "smiles":
        if input_smiles_type == "canonical":
            inp = Chem.MolToSmiles(mol, canonical=True)
        elif input_smiles_type == "random":
            randomized_smiles = randomize_smiles(mol)
            if randomized_smiles is None:
                return None
            inp = randomized_smiles
        else:
            raise ValueError(f"Invalid input_smiles_type: {input_smiles_type}")
    elif input_smiles_syntax == "crisp_smiles":
        smiles = Chem.MolToSmiles(mol)
        inp = CRISPSmilesConverter.encode(
            smiles, deferred=crisp_input_deferred, smiles_mode=input_smiles_type
        )
    else:
        raise ValueError(f"Invalid input_smiles_syntax: {input_smiles_syntax}")

    return inp


def process_translation(
    smiles: str,
    smiles_preprocessing_inp: Optional[SmilesPreprocessing],
    smiles_preprocessing_out: Optional[SmilesPreprocessing],
    max_atoms: int = 150,
) -> Optional[Tuple[str, str]]:
    """
    Process a single SMILES string into an (input, output) pair.

    Args:
        smiles (str): SMILES string.
        smiles_preprocessing_inp (SmilesPreprocessing): Input smiles preprocessing configuration.
        smiles_preprocessing_out (SmilesPreprocessing): Output smiles preprocessing configuration.
        max_atoms (int): Skip molecules with more atoms than this.

    Returns:
        Optional[Tuple[str, str]]: (input_smiles, output_smiles) or None if skipped.
    """
    if not smiles_preprocessing_inp:
        return None
    if not smiles_preprocessing_out:
        return None

    inp = process_single_smiles(smiles, smiles_preprocessing_inp, max_atoms)
    out = process_single_smiles(smiles, smiles_preprocessing_out, max_atoms)
    if inp is None or out is None:
        return None
    return (inp, out)


def process_reaction(
    rxn_smiles: str,
    smiles_preprocessing_reactants: Optional[SmilesPreprocessing],
    smiles_preprocessing_products: Optional[SmilesPreprocessing],
    max_atoms: int = 150,
) -> Optional[Tuple[str, str]]:
    """
    Process a reaction SMILES string into a (reactants, products) pair.

    Args:
        rxn_smiles (str): Reaction SMILES in "reactants>>products" format.
        augment (bool): If True, randomize individual molecule SMILES.
        max_atoms (int): Skip molecules with more atoms than this.

    Returns:
        Optional[Tuple[str, str]]: (reactant_smiles, product_smiles) or None if invalid.
    """
    if ">>" not in rxn_smiles:
        return None
    if not smiles_preprocessing_reactants:
        return None
    if not smiles_preprocessing_products:
        return None

    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        return None

    reactants_str, products_str = parts

    def _process_side(smiles_str, smiles_preprocessing, max_atoms):
        pieces = []
        for smi in smiles_str.split("."):
            processed = process_single_smiles(
                smi.strip(), smiles_preprocessing, max_atoms
            )
            if processed is None:
                return None
            pieces.append(processed)
        if any(p is None for p in pieces):
            return None
        return ".".join(pieces)

    reactants = _process_side(reactants_str, smiles_preprocessing_reactants, max_atoms)
    products = _process_side(products_str, smiles_preprocessing_products, max_atoms)

    if reactants is None or products is None:
        return None
    return (reactants, products)


def generate_training_data(
    data: List,
    mode: str = "translation",
    input_smiles_preprocessing: SmilesPreprocessing | None = None,
    output_smiles_preprocessing: SmilesPreprocessing | None = None,
    max_atoms: int = 150,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Generate training data pairs based on the preprocessing mode.

    Args:
        data (List): Input data. For MODE_RANDOM_TO_CANONICAL and MODE_RANDOM_TO_RANDOM,
            a list of RDKit Mol objects. For MODE_REACTION, a list of reaction SMILES
            strings in "reactants>>products" format.
        mode (str): Preprocessing mode. One of MODE_RANDOM_TO_CANONICAL,
            MODE_RANDOM_TO_RANDOM, or MODE_REACTION.
        max_atoms (int): Maximum number of atoms allowed per molecule.
        verbose (bool): Print progress information.

    Returns:
        List[Tuple[str, str]]: List of (input_smiles, output_smiles) pairs.
    """
    all_data: List[Tuple[str, str]] = []
    skipped = 0

    if verbose:
        print(f"Generating data (mode={mode}, n={len(data)})")

    for i, item in enumerate(data):
        if verbose and (i + 1) % 1000 == 0:
            print(
                f"  Processed {i + 1}/{len(data)} → {len(all_data)} samples",
                end="\r",
            )

        if mode == "reaction":
            result = process_reaction(
                item,
                input_smiles_preprocessing,
                output_smiles_preprocessing,
                max_atoms=max_atoms,
            )
        elif mode == "translation":
            result = process_translation(
                item,
                input_smiles_preprocessing,
                output_smiles_preprocessing,
                max_atoms=max_atoms,
            )

        if result is not None:
            all_data.append(result)
        else:
            skipped += 1

    if verbose:
        print(
            f"\n✓ Generated {len(all_data)} samples from {len(data)} items (skipped: {skipped})"
        )

    return all_data


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC DATASET WITH AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════


class DynamicSmilesDataset(Dataset):
    """
    Dataset that regenerates data each epoch for augmentation.

    For MODE_RANDOM_TO_CANONICAL and MODE_RANDOM_TO_RANDOM, pass RDKit Mol objects.
    For MODE_REACTION, pass reaction SMILES strings.
    """

    def __init__(
        self,
        data: List,
        mode: str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 512,
        input_smiles_preprocessing: SmilesPreprocessing | None = None,
        output_smiles_preprocessing: SmilesPreprocessing | None = None,
        max_atoms: int = 150,
        regenerate_per_epoch: bool = True,
    ):
        if not data:
            raise ValueError("data list cannot be empty")

        self.data_source = data
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.input_smiles_preprocessing = input_smiles_preprocessing
        self.output_smiles_preprocessing = output_smiles_preprocessing
        self.max_atoms = max_atoms
        self.regenerate_per_epoch = regenerate_per_epoch

        # Generate initial data
        print("Generating initial training data...")
        self.data = generate_training_data(
            self.data_source,
            mode=self.mode,
            input_smiles_preprocessing=self.input_smiles_preprocessing,
            output_smiles_preprocessing=self.output_smiles_preprocessing,
            max_atoms=self.max_atoms,
        )

        if not self.data:
            raise ValueError(
                f"No valid training samples generated from {len(self.data_source)} items. "
                "Check your data and max_atoms parameter."
            )

        print(f"✓ Dataset initialized with {len(self.data)} samples\n")

    def regenerate_data(self):
        """Regenerate the dataset with new random SMILES representations."""
        if self.regenerate_per_epoch:
            print("\nRegenerating training data with new SMILES augmentation...")
            old_size = len(self.data)
            self.data = generate_training_data(
                self.data_source,
                mode=self.mode,
                input_smiles_preprocessing=self.input_smiles_preprocessing,
                output_smiles_preprocessing=self.output_smiles_preprocessing,
                max_atoms=self.max_atoms,
                verbose=True,
            )
            if not self.data:
                raise ValueError(f"Data regeneration failed! Previous size: {old_size}")
            print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Sanity check
        if idx >= len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )

        item = self.data[idx]

        # Verify item is a tuple
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"Expected tuple of (input_smiles, output_bonds) at index {idx}, "
                f"but got {type(item)}. This suggests a data generation error."
            )

        input_smiles, output_bonds = item

        input_encoding = self.tokenizer(
            input_smiles,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output_encoding = self.tokenizer(
            output_bonds,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = output_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StaticSmilesDataset(Dataset):
    """
    Standard dataset for validation/test (no augmentation).

    Pass pre-processed tuples of (input_smiles, output_smiles).
    """

    def __init__(
        self,
        data: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 512,
        input_smiles_preprocessing: SmilesPreprocessing | None = None,
        output_smiles_preprocessing: SmilesPreprocessing | None = None,
        max_atoms: int = 150,
    ):
        if not data:
            raise ValueError("data list cannot be empty")

        if not isinstance(data[0], tuple):
            raise TypeError(
                "StaticSmilesDataset expects tuples of (input_smiles, output_smiles). "
                "Use DynamicSmilesDataset for raw data with augmentation."
            )

        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.input_smiles_preprocessing = input_smiles_preprocessing
        self.output_smiles_preprocessing = output_smiles_preprocessing
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_smiles, output_bonds = self.data[idx]

        input_encoding = self.tokenizer(
            input_smiles,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output_encoding = self.tokenizer(
            output_bonds,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = output_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataRegenerationCallback(TrainerCallback):
    """Callback to regenerate training data at the beginning of each epoch"""

    def __init__(self, train_dataset: DynamicSmilesDataset):
        self.train_dataset = train_dataset
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        new_epoch = int(state.epoch) if state.epoch is not None else 0

        if new_epoch > self.current_epoch:
            self.current_epoch = new_epoch
            if new_epoch > 0:  # Don't regenerate before first epoch
                print(f"\n{'=' * 60}")
                print(f"Starting Epoch {new_epoch + 1}")
                self.train_dataset.regenerate_data()
                print(f"{'=' * 60}\n")


class TrainingAnalysisCallback(TrainerCallback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_loss_path = os.path.join(self.save_dir, "train_loss_by_step.txt")
        self.eval_loss_path = os.path.join(self.save_dir, "eval_loss_by_step.txt")

    def _append_line(self, path: str, line: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        if not logs:
            return

        if "loss" in logs:
            self._append_line(
                self.train_loss_path,
                f"{state.global_step}\t{logs['loss']}",
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_local_process_zero:
            return
        if not metrics:
            return

        if "eval_loss" in metrics:
            step = state.global_step
            eval_loss = metrics["eval_loss"]
            self._append_line(self.eval_loss_path, f"{step}\t{eval_loss}")

            per_eval_path = os.path.join(self.save_dir, f"eval_loss_step_{step}.txt")
            with open(per_eval_path, "w", encoding="utf-8") as f:
                f.write(f"global_step\t{step}\n")
                f.write(f"eval_loss\t{eval_loss}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CustomSmallConfig:
    """Configuration for custom tiny transformer"""

    vocab_size: int | None = None
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    max_input_length: int = 256
    max_output_length: int = 256
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    warmup_steps: int = 8000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.998
    adam_epsilon: float = 1e-9
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    save_total_limit: int = 3
    eval_steps: int = 1000
    save_steps: int = 1000
    logging_steps: int = 100
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001
    output_dir: str = "/content/drive/MyDrive/custom_tiny_model"
    logging_dir: str = "/content/drive/MyDrive/logs"
    analysis_dir: str | None = None
    num_beams: int = 5
    length_penalty: float = 1.0
    early_stopping_generation: bool = True


class CustomSmallModelBuilder:
    """Build a custom small BART transformer from scratch"""

    @staticmethod
    def build_model(config: CustomSmallConfig, tokenizer: PreTrainedTokenizer):
        vocab_size = len(tokenizer) + 1

        bart_config = BartConfig(
            vocab_size=vocab_size,
            d_model=config.hidden_size,
            encoder_layers=config.num_hidden_layers,
            decoder_layers=config.num_hidden_layers,
            encoder_attention_heads=config.num_attention_heads,
            decoder_attention_heads=config.num_attention_heads,
            encoder_ffn_dim=config.intermediate_size,
            decoder_ffn_dim=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            attention_dropout=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            activation_function="gelu",
            init_std=0.02,
            classifier_dropout=0.0,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=13,
            eos_token_id=14,
            decoder_start_token_id=13,
            forced_eos_token_id=14,
        )

        model = BartForConditionalGeneration(bart_config)

        model.generation_config = GenerationConfig(
            max_length=config.max_output_length,
            num_beams=config.num_beams,
            length_penalty=config.length_penalty,
            early_stopping=config.early_stopping_generation,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=13,
            eos_token_id=14,
            decoder_start_token_id=13,
            forced_eos_token_id=14,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(
            f"✓ BART model created with {num_params:,} parameters ({num_params / 1e6:.2f}M)"
        )

        return model


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER WITH AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════


class SmilesTrainer:
    """Trainer with built-in SMILES augmentation per epoch."""

    def __init__(
        self,
        config: CustomSmallConfig,
        tokenizer: PreTrainedTokenizer,
        train_data: List,
        mode: str = "translation",
        val_data: Optional[List] = None,
        test_data: Optional[List[Tuple[str, str]]] = None,
        max_atoms: int = 150,
        regenerate_per_epoch: bool = True,
        analysis_dir: str | None = None,
        input_smiles_preprocessing: SmilesPreprocessing | None = None,
        output_smiles_preprocessing: SmilesPreprocessing | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            config (CustomSmallConfig): Model and training configuration.
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            train_data (List): Training data. SMILES strings for translation mode,
                reaction SMILES strings for reaction mode.
            mode (str): Preprocessing mode ("translation" or "reaction").
            val_data (Optional[List]): Validation data. Either raw data (same format as
                train_data, will be processed once) or pre-processed tuples.
            test_data (Optional[List[Tuple[str, str]]]): Pre-processed test tuples.
            max_atoms (int): Maximum atoms per molecule.
            regenerate_per_epoch (bool): Re-randomize SMILES each epoch.
            analysis_dir (str | None): Directory for analysis outputs.
            input_smiles_preprocessing (SmilesPreprocessing | None): Input preprocessing config.
            output_smiles_preprocessing (SmilesPreprocessing | None): Output preprocessing config.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode
        self.trainer = None

        self.analysis_dir = (
            analysis_dir
            if analysis_dir is not None
            else (
                self.config.analysis_dir
                if self.config.analysis_dir is not None
                else os.path.join(self.config.output_dir, "analysis")
            )
        )
        os.makedirs(self.analysis_dir, exist_ok=True)

        self.input_smiles_preprocessing = input_smiles_preprocessing
        self.output_smiles_preprocessing = output_smiles_preprocessing

        # Create dynamic training dataset that regenerates each epoch
        print("=" * 60)
        print(f"INITIALIZING TRAINING DATASET (mode={mode})")
        print("=" * 60)
        self.train_dataset = DynamicSmilesDataset(
            data=train_data,
            mode=mode,
            tokenizer=tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            input_smiles_preprocessing=input_smiles_preprocessing,
            output_smiles_preprocessing=output_smiles_preprocessing,
            max_atoms=max_atoms,
            regenerate_per_epoch=regenerate_per_epoch,
        )

        # Validation dataset - accepts raw data or pre-processed tuples
        self.val_dataset = None
        if val_data is not None:
            if isinstance(val_data[0], tuple):
                self.val_dataset = StaticSmilesDataset(
                    val_data,
                    tokenizer,
                    config.max_input_length,
                    config.max_output_length,
                )
            else:
                print("=" * 60)
                print("INITIALIZING VALIDATION DATASET")
                print("=" * 60)
                processed_val = generate_training_data(
                    val_data,
                    mode=mode,
                    input_smiles_preprocessing=input_smiles_preprocessing,
                    output_smiles_preprocessing=output_smiles_preprocessing,
                    max_atoms=max_atoms,
                    verbose=True,
                )
                if processed_val:
                    self.val_dataset = StaticSmilesDataset(
                        processed_val,
                        tokenizer,
                        config.max_input_length,
                        config.max_output_length,
                    )
                print()

        self.test_dataset = (
            StaticSmilesDataset(
                test_data, tokenizer, config.max_input_length, config.max_output_length
            )
            if test_data
            else None
        )

        print("Building custom BART model from scratch...")
        self.model = CustomSmallModelBuilder.build_model(config, tokenizer)

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if (
            self.trainer is not None
            and getattr(self, "analysis_dir", None) is not None
            and self.trainer.is_world_process_zero()
        ):
            step = self.trainer.state.global_step
            gen_path = os.path.join(
                self.analysis_dir, f"eval_generations_step_{step}.txt"
            )
            with open(gen_path, "w", encoding="utf-8") as f:
                for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
                    f.write(f"{i}\t{pred}\t{label}\n")

        exact_matches = [
            pred == label for pred, label in zip(decoded_preds, decoded_labels)
        ]
        exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

        correct_tokens = 0
        total_tokens = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = pred.split()
            label_tokens = label.split()
            min_len = min(len(pred_tokens), len(label_tokens))
            correct_tokens += sum(
                pred == label
                for pred, label in zip(pred_tokens[:min_len], label_tokens[:min_len])
            )
            total_tokens += len(label_tokens)
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(
            f"  Exact match  : {exact_match:.4f}  ({sum(exact_matches)}/{len(exact_matches)})"
        )
        print(f"  Token acc    : {token_accuracy:.4f}")
        if decoded_preds:
            print(f"  Example pred : {decoded_preds[0]}")
            print(f"  Example label: {decoded_labels[0]}")
        print(f"{'=' * 60}\n")

        return {
            "exact_match": exact_match,
            "token_accuracy": token_accuracy,
        }

    def train(self):
        """Train the model with data augmentation"""

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self.val_dataset else None,
            eval_strategy="steps" if self.val_dataset else "no",
            save_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            report_to=["tensorboard"],
            push_to_hub=False,
            predict_with_generate=True,
            generation_max_length=self.config.max_output_length,
            generation_num_beams=self.config.num_beams,
        )

        # Add data regeneration callback
        callbacks = [
            DataRegenerationCallback(self.train_dataset),
            TrainingAnalysisCallback(self.analysis_dir),
        ]

        if self.val_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        print("🚀 Starting training with SMILES augmentation per epoch...")
        self.trainer.train()

        final_model_path = os.path.join(self.config.output_dir, "final_model")
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"✓ Model saved to {final_model_path}")

        return self.trainer

    def predict(
        self,
        input_smiles: List[str],
        model_path: str | None = None,
        batch_size: int = 8,
        num_beams: int | None = None,
        return_scores: bool = False,
        debug: bool = False,
    ):
        """Run inference on a list of SMILES strings"""
        if model_path:
            model = BartForConditionalGeneration.from_pretrained(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            model = self.model

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(
            f"  Device: {device}  |  Samples: {len(input_smiles)}  |  Batch size: {batch_size}"
        )

        num_beams = num_beams if num_beams is not None else self.config.num_beams

        gen_config = GenerationConfig(
            max_length=self.config.max_output_length,
            num_beams=num_beams,
            length_penalty=self.config.length_penalty,
            early_stopping=self.config.early_stopping_generation,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=13,
            eos_token_id=14,
            decoder_start_token_id=13,  # Must match BOS
            forced_eos_token_id=14,
        )

        # DEBUG: Print generation config
        if debug:
            print("\n" + "=" * 60)
            print("GENERATION CONFIG DEBUG:")
            print(f"  pad_token_id: {gen_config.pad_token_id}")
            print(f"  bos_token_id: {gen_config.bos_token_id}")
            print(f"  eos_token_id: {gen_config.eos_token_id}")
            print(f"  decoder_start_token_id: {gen_config.decoder_start_token_id}")
            print("=" * 60 + "\n")

        predictions: List[str] = []
        scores: List[float] = []

        for i in range(0, len(input_smiles), batch_size):
            batch = input_smiles[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=self.config.max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    return_dict_in_generate=return_scores,
                    output_scores=return_scores,
                )

            if return_scores:
                sequences = outputs.sequences
                scores.extend(outputs.sequences_scores.cpu().tolist())
            else:
                sequences = outputs

            # DEBUG: Show first batch details
            if debug and i == 0:
                print("\n" + "=" * 60)
                print("FIRST BATCH TOKEN IDs DEBUG:")
                print(f"  Generated sequence shape: {sequences.shape}")
                print(f"  First sequence token IDs: {sequences[0].cpu().tolist()}")

                # Decode WITH special tokens to see everything
                debug_decode_with = self.tokenizer.decode(
                    sequences[0], skip_special_tokens=False
                )
                debug_decode_without = self.tokenizer.decode(
                    sequences[0], skip_special_tokens=True
                )

                print("\n  Decoded WITH special tokens:")
                print(f"    '{debug_decode_with}'")
                print("  Decoded WITHOUT special tokens:")
                print(f"    '{debug_decode_without}'")
                print("=" * 60 + "\n")

            batch_preds = self.tokenizer.batch_decode(
                sequences, skip_special_tokens=True
            )
            predictions.extend([p.strip() for p in batch_preds])
            print(
                f"  Processed {min(i + batch_size, len(input_smiles))}/{len(input_smiles)}",
                end="\r",
            )

        print()

        if return_scores:
            return predictions, scores
        return predictions

    def evaluate_predictions(
        self,
        input_smiles: List[str],
        target_smiles: List[str],
        model_path: str | None = None,
        batch_size: int = 8,
        debug: bool = True,  # Enable debug by default
    ):
        """Run inference and print a full accuracy report"""

        # DEBUG: Show what we're comparing against
        if debug:
            print("\n" + "=" * 60)
            print("TARGET EXAMPLES DEBUG:")
            for i, tgt in enumerate(target_smiles[:3]):
                print(f"  Target {i}: '{tgt}'")
                print(f"    Length: {len(tgt)}")
                print(f"    Repr: {repr(tgt)}")
            print("=" * 60 + "\n")

        predictions = self.predict(
            input_smiles, model_path=model_path, batch_size=batch_size, debug=debug
        )

        # DEBUG: Show predictions BEFORE comparison
        if debug:
            print("\n" + "=" * 60)
            print("PREDICTION EXAMPLES DEBUG:")
            for i, pred in enumerate(predictions[:3]):
                print(f"  Prediction {i}: '{pred}'")
                print(f"    Length: {len(pred)}")
                print(f"    Repr: {repr(pred)}")
            print("=" * 60 + "\n")

        exact_matches = [p == t for p, t in zip(predictions, target_smiles)]
        exact_match = sum(exact_matches) / len(exact_matches)

        correct_tokens = 0
        total_tokens = 0
        for pred, label in zip(predictions, target_smiles):
            pt = pred.split()
            lt = label.split()
            correct_tokens += sum(pred == label for pred, label in zip(pt, lt))
            total_tokens += len(lt)
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"  Samples              : {len(input_smiles)}")
        print(
            f"  Exact match accuracy : {exact_match:.4f}  ({sum(exact_matches)}/{len(exact_matches)})"
        )
        print(f"  Token-level accuracy : {token_accuracy:.4f}")
        print(f"{'=' * 60}")
        print("\nSAMPLE PREDICTIONS (first 5):")
        print(f"{'=' * 60}")
        for i, (inp, pred, tgt, ok) in enumerate(
            zip(input_smiles[:5], predictions[:5], target_smiles[:5], exact_matches[:5])
        ):
            status = "✓" if ok else "✗"
            print(f"\n[{i}] {status} Match: {ok}")
            print(f"  Input : {inp}")
            print(f"  Pred  : '{pred}'")
            print(f"  Target: '{tgt}'")
            if not ok and debug:
                # Character-by-character comparison
                print(f"  Pred repr  : {repr(pred)}")
                print(f"  Target repr: {repr(tgt)}")
                print(f"  Pred tokens  : {pred.split()}")
                print(f"  Target tokens: {tgt.split()}")
        print(f"\n{'=' * 60}\n")

        return {
            "exact_match": exact_match,
            "token_accuracy": token_accuracy,
            "predictions": predictions,
            "exact_matches": exact_matches,
        }
