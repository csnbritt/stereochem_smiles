import re

from rdkit import Chem

STEREO_R_ISOTOPE = 10001
STEREO_S_ISOTOPE = 10002
STEREO_R_TOKEN = "[STEREO_R]"
STEREO_S_TOKEN = "[STEREO_S]"

STEREO_E_ISOTOPE = 10003
STEREO_Z_ISOTOPE = 10004
STEREO_E_TOKEN = "[STEREO_E]"
STEREO_Z_TOKEN = "[STEREO_Z]"


class CRISPSmiles:
    """
    A class for converting between standard SMILES notation and CRISP SMILES notation.
    """

    def __init__(self):
        pass

    def encode(
        self,
        smiles: str,
        deferred: bool = True,
        smiles_mode: str = "canonical",
    ) -> str:
        """
        Convert a SMILES string to the new CRISP syntax.

        Processes both tetrahedral (R/S) and double-bond (E/Z) stereochemistry.
        Tetrahedral stereo is handled first, then double-bond stereo.

        Args:
            smiles (str): Input SMILES string.
            deferred (bool): If True, use deferred syntax with assignment blocks.
                If False, use inline stereo tokens.
            smiles_mode (str): "canonical" or "random". Determines whether to generate
                canonical or random SMILES.

        Returns:
            str: SMILES string with CRISP stereo syntax.
        """
        # Step 1: Replace @/@@ with R/S isotopes and /\ with E/Z isotopes.
        # Both are handled in a single MolFromSmiles→MolToSmiles pass to preserve
        # the input atom order (avoids re-parse canonicalization).
        smiles = self._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode=smiles_mode
        )

        # Step 2: Convert isotope labels to tokens (inline or deferred).
        if deferred:
            smiles = self._convert_syntax_deferred(smiles)
            smiles = self._convert_db_syntax_deferred(smiles)
        else:
            smiles = self._convert_syntax_non_deferred(smiles)
            smiles = self._convert_db_syntax_non_deferred(smiles)

        return smiles

    def decode(self, new_syntax: str, deferred: bool = True) -> str:
        """
        Convert a CRISP syntax string back to a regular SMILES string.

        Reverses both double-bond (E/Z) and tetrahedral (R/S) encoding.
        Double-bond stereo is reverted first, then tetrahedral stereo.

        Args:
            new_syntax (str): SMILES string with CRISP stereo syntax.
            deferred (bool): If True, parse deferred assignment blocks.
                If False, parse inline stereo tokens.

        Returns:
            str: Standard SMILES string with conventional stereochemistry notation.
        """
        # Step 1: Text-level inversion — replace tokens/placeholders with isotopes.
        # Both steps are pure string operations, so order doesn't matter here.
        if deferred:
            new_syntax = self._invert_db_syntax_deferred(new_syntax)
            new_syntax = self._invert_syntax_deferred(new_syntax)
        else:
            new_syntax = self._invert_db_syntax_non_deferred(new_syntax)
            new_syntax = self._invert_syntax_non_deferred(new_syntax)

        # Step 2: RDKit revert — read isotopes, restore real stereo, produce SMILES.
        # At this point new_syntax has both R/S and E/Z isotopes and is valid SMILES.
        restored_smiles = self._revert_all_stereochem_to_global(new_syntax)
        return restored_smiles

    def _generate_local_stereochem_syntax_for_mol(
        self, smiles: str, smiles_mode: str = "canonical"
    ) -> str:
        """
        Generate local stereochemistry syntax for a molecule.

        Encodes both tetrahedral (R/S) and double-bond (E/Z) stereochemistry as
        isotope labels in a single MolFromSmiles → MolToSmiles pass so that the
        input atom order is preserved (no re-parse canonicalization).

        Args:
            smiles (str): Input SMILES string.
            smiles_mode (str): "canonical" or "random". Determines whether to generate
                canonical or random SMILES.

        Returns:
            str: SMILES string with R/S and E/Z encoded as isotope labels, with
                @/@@ and /\\ notation removed.
        """
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # ── Tetrahedral (R/S) ────────────────────────────────────────────────
        for atom in mol.GetAtoms():
            if atom.GetChiralTag():
                cip_code = atom.GetProp("_CIPCode")  # 'R' or 'S'
                atom.SetIsotope(
                    STEREO_R_ISOTOPE if cip_code == "R" else STEREO_S_ISOTOPE
                )
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

        # ── Double-bond (E/Z) ────────────────────────────────────────────────
        # Collect absolute E/Z via rdCIPLabeler (sets _CIPCode = "E"/"Z" on bonds).
        # Prefer marking the begin atom; fall back to end atom if begin already
        # carries an R/S isotope (avoids overwriting tetrahedral stereo).
        Chem.rdCIPLabeler.AssignCIPLabels(mol)

        db_targets = {}
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0 and bond.HasProp("_CIPCode"):
                cip = bond.GetProp("_CIPCode")  # "E" or "Z"
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                mark_idx = begin_idx
                # TODO: will this work for molecules with e/z bonds and r/s stereochem on both ends?
                if mol.GetAtomWithIdx(begin_idx).GetIsotope() in (
                    STEREO_R_ISOTOPE,
                    STEREO_S_ISOTOPE,
                ):
                    mark_idx = end_idx
                db_targets[mark_idx] = cip

        for bond in mol.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

        for idx, cip in db_targets.items():
            isotope = STEREO_E_ISOTOPE if cip == "E" else STEREO_Z_ISOTOPE
            mol.GetAtomWithIdx(idx).SetIsotope(isotope)

        # ── Single MolToSmiles pass ──────────────────────────────────────────
        if smiles_mode == "canonical":
            smiles = Chem.MolToSmiles(mol, canonical=True)
        elif smiles_mode == "random":
            smiles = Chem.MolToSmiles(mol, doRandom=True)
        else:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        smiles = smiles.replace("/", "").replace("\\", "")
        return smiles

    def _revert_all_stereochem_to_global(self, smiles: str) -> str:
        """
        Restore both tetrahedral (R/S) and double-bond (E/Z) stereochemistry from
        isotope labels in a single RDKit pass.

        Args:
            smiles (str): SMILES with R/S and/or E/Z isotope labels.

        Returns:
            str: Standard SMILES with all stereochemistry restored.
        """
        mol = Chem.MolFromSmiles(smiles)

        # ── Tetrahedral R/S ──────────────────────────────────────────────────────
        chiral_targets = {}
        for atom in mol.GetAtoms():
            iso = atom.GetIsotope()
            if iso == STEREO_R_ISOTOPE:
                chiral_targets[atom.GetIdx()] = "R"
            elif iso == STEREO_S_ISOTOPE:
                chiral_targets[atom.GetIdx()] = "S"

        # ── Double-bond E/Z ──────────────────────────────────────────────────────
        db_targets = {}
        for atom in mol.GetAtoms():
            iso = atom.GetIsotope()
            if iso == STEREO_E_ISOTOPE:
                db_targets[atom.GetIdx()] = "E"
            elif iso == STEREO_Z_ISOTOPE:
                db_targets[atom.GetIdx()] = "Z"

        # Strip all stereo isotopes
        for atom in mol.GetAtoms():
            if atom.GetIsotope() in (
                STEREO_R_ISOTOPE,
                STEREO_S_ISOTOPE,
                STEREO_E_ISOTOPE,
                STEREO_Z_ISOTOPE,
            ):
                atom.SetIsotope(0)

        # Restore tetrahedral stereo (trial-and-flip)
        for idx in chiral_targets:
            mol.GetAtomWithIdx(idx).SetChiralTag(
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
            )
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for idx, target_cip in chiral_targets.items():
            atom = mol.GetAtomWithIdx(idx)
            if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != target_cip:
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Restore double-bond stereo (trial-and-flip via FindPotentialStereo)
        db_bonds = {}
        saved_stereo_atoms = {}
        for idx in db_targets:
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    begin_neighbors = [
                        n.GetIdx()
                        for n in mol.GetAtomWithIdx(begin_idx).GetNeighbors()
                        if n.GetIdx() != end_idx
                    ]
                    end_neighbors = [
                        n.GetIdx()
                        for n in mol.GetAtomWithIdx(end_idx).GetNeighbors()
                        if n.GetIdx() != begin_idx
                    ]
                    if begin_neighbors and end_neighbors:
                        ref_begin = begin_neighbors[0]
                        ref_end = end_neighbors[0]
                        bond.SetStereoAtoms(ref_begin, ref_end)
                        bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
                        db_bonds[idx] = bond.GetIdx()
                        saved_stereo_atoms[bond.GetIdx()] = (ref_begin, ref_end)
                    break

        # Check absolute E/Z via SMILES round-trip and flip if wrong.
        # rdCIPLabeler.AssignCIPLabels corrupts the mol's internal stereo
        # representation, so CIP must be checked on a fresh molecule obtained
        # by writing + re-parsing SMILES.  Each bond is probed individually
        # (only its stereo is set) so the single CIP-labelled bond in the
        # probe mol unambiguously corresponds to the bond under test.
        flip_needed = {}
        for idx in db_targets:
            if idx not in db_bonds:
                continue
            bond_idx = db_bonds[idx]
            ref_begin, ref_end = saved_stereo_atoms[bond_idx]
            # Clear all E/Z stereo so only one bond is active in the probe
            for other_bond_idx in db_bonds.values():
                mol.GetBondWithIdx(other_bond_idx).SetStereo(
                    Chem.rdchem.BondStereo.STEREONONE
                )
            mol.GetBondWithIdx(bond_idx).SetStereoAtoms(ref_begin, ref_end)
            mol.GetBondWithIdx(bond_idx).SetStereo(Chem.rdchem.BondStereo.STEREOE)
            probe_smi = Chem.MolToSmiles(mol, canonical=True)
            probe_mol = Chem.MolFromSmiles(probe_smi)
            Chem.rdCIPLabeler.AssignCIPLabels(probe_mol)
            probe_cip = None
            for pb in probe_mol.GetBonds():
                if pb.HasProp("_CIPCode"):
                    probe_cip = pb.GetProp("_CIPCode")
                    break
            flip_needed[idx] = probe_cip != db_targets[idx]

        # Apply the determined stereo to all bonds at once
        for idx in db_targets:
            if idx not in db_bonds:
                continue
            bond_idx = db_bonds[idx]
            ref_begin, ref_end = saved_stereo_atoms[bond_idx]
            mol.GetBondWithIdx(bond_idx).SetStereoAtoms(ref_begin, ref_end)
            if flip_needed[idx]:
                mol.GetBondWithIdx(bond_idx).SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            else:
                mol.GetBondWithIdx(bond_idx).SetStereo(Chem.rdchem.BondStereo.STEREOE)

        return Chem.MolToSmiles(mol, canonical=False)

    def _convert_syntax_non_deferred(self, smiles: str) -> str:
        """
        Convert a SMILES string with local stereochemistry syntax to a regular SMILES string.

        Args:
            smiles: SMILES string with local stereochemistry syntax

        Returns:
            SMILES string with regular syntax
        """
        for ele in re.findall(r"\[(.*?)\]", smiles):
            iso_match = re.match(r"^(\d+)(.*)", ele)
            if not iso_match:
                continue
            isotope = int(iso_match.group(1))
            if isotope in (STEREO_R_ISOTOPE, STEREO_S_ISOTOPE):
                original_token = "[" + ele + "]"
                atom_part = iso_match.group(2)
                token_no_isotope = "[" + atom_part + "]"
                stereo_token = (
                    STEREO_R_TOKEN if isotope == STEREO_R_ISOTOPE else STEREO_S_TOKEN
                )
                smiles = smiles.replace(original_token, token_no_isotope + stereo_token)
        return smiles

    def _invert_syntax_non_deferred(self, converted_smiles: str) -> str:
        """
        Invert the syntax of a SMILES string with local stereochemistry syntax.

        Args:
            converted_smiles: SMILES string with local stereochemistry syntax

        Returns:
            SMILES string with inverted syntax
        """
        converted_smiles = re.sub(
            r"\[([^\]]+)\]\[STEREO_R\]",
            lambda m: "[" + str(STEREO_R_ISOTOPE) + m.group(1) + "]",
            converted_smiles,
        )
        converted_smiles = re.sub(
            r"\[([^\]]+)\]\[STEREO_S\]",
            lambda m: "[" + str(STEREO_S_ISOTOPE) + m.group(1) + "]",
            converted_smiles,
        )
        return converted_smiles

    def _revert_local_stereochem_to_global_non_deferred(
        self, inverted_syntax_smiles: str
    ) -> str:
        """
        Revert a SMILES string with local stereochemistry syntax back to a regular SMILES string.

        Args:
            inverted_syntax_smiles: SMILES string with local stereochemistry syntax

        Returns:
            SMILES string with regular syntax
        """
        mol = Chem.MolFromSmiles(inverted_syntax_smiles)

        # Read target R/S from isotopes
        targets = {}
        for atom in mol.GetAtoms():
            isotope = atom.GetIsotope()
            if isotope == STEREO_R_ISOTOPE:
                targets[atom.GetIdx()] = "R"
            elif isotope == STEREO_S_ISOTOPE:
                targets[atom.GetIdx()] = "S"

        # Strip isotopes so CIP runs on a clean molecule
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)

        # Set CW for all, then check and flip where needed
        for idx in targets:
            mol.GetAtomWithIdx(idx).SetChiralTag(
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
            )
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for idx, target_cip in targets.items():
            atom = mol.GetAtomWithIdx(idx)
            if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != target_cip:
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        return Chem.MolToSmiles(mol, canonical=False)

    def _convert_syntax_deferred(self, smiles: str) -> str:
        """
        Convert a SMILES string with local stereochemistry syntax to a deferred syntax SMILES string.

        Args:
            smiles: SMILES string with local stereochemistry syntax

        Returns:
            SMILES string with deferred syntax
        """
        assignments = []
        counter = [0]

        def _replace_isotope(match):
            isotope = int(match.group(1))
            if isotope not in (STEREO_R_ISOTOPE, STEREO_S_ISOTOPE):
                return match.group(0)
            counter[0] += 1
            atom_part = match.group(2)
            stereo_token = (
                STEREO_R_TOKEN if isotope == STEREO_R_ISOTOPE else STEREO_S_TOKEN
            )
            assignments.append((counter[0], stereo_token))
            return f"[{atom_part}][STEREO_CENTER_{counter[0]}]"

        smiles = re.sub(r"\[(\d+)(.*?)\]", _replace_isotope, smiles)

        if assignments:
            assignment_block = "|" + "".join(
                f"[STEREO_CENTER_{num}]{token}" for num, token in assignments
            )
            smiles += assignment_block

        return smiles

    def _invert_syntax_deferred(self, converted_smiles: str) -> str:
        """
        Invert the deferred tetrahedral (R/S) syntax back to isotope labels.

        Only consumes [STEREO_CENTER_N][STEREO_R/S] assignments from the | block.
        Any other deferred assignments (e.g. [DB_STEREO_N][STEREO_E/Z]) are preserved.

        Args:
            converted_smiles (str): SMILES with deferred tetrahedral syntax.

        Returns:
            str: SMILES with R/S isotope labels and any remaining | block intact.
        """
        if "|" not in converted_smiles:
            return converted_smiles

        graph_part, assignment_block = converted_smiles.split("|", 1)

        # Parse STEREO_CENTER assignment pairs
        assignments = {}
        for match in re.finditer(
            r"\[STEREO_CENTER_(\d+)\]\[(STEREO_[RS])\]", assignment_block
        ):
            center_num = int(match.group(1))
            config = match.group(2)
            isotope = STEREO_R_ISOTOPE if config == "STEREO_R" else STEREO_S_ISOTOPE
            assignments[center_num] = isotope

        # Remove consumed assignments from block
        remaining = re.sub(
            r"\[STEREO_CENTER_\d+\]\[STEREO_[RS]\]", "", assignment_block
        )

        # Replace placeholders in graph part with isotopes
        for center_num, isotope in assignments.items():
            pattern = r"\[([^\]]+)\]\[STEREO_CENTER_" + str(center_num) + r"\]"

            def _replace_center(match: re.Match[str], iso: int = isotope) -> str:
                return "[" + str(iso) + match.group(1) + "]"

            graph_part = re.sub(pattern, _replace_center, graph_part)

        # Reconstruct with any remaining assignment content
        remaining_parts = [p for p in remaining.split("|") if p.strip()]
        if remaining_parts:
            return graph_part + "|" + "|".join(remaining_parts)
        return graph_part

    def _revert_local_stereochem_to_global_deferred(
        self, inverted_syntax_smiles: str
    ) -> str:
        """
        Revert a SMILES string with deferred stereochemistry syntax back to a regular SMILES string.

        Args:
            inverted_syntax_smiles: SMILES string with deferred stereochemistry syntax

        Returns:
            SMILES string with regular syntax
        """
        mol = Chem.MolFromSmiles(inverted_syntax_smiles)

        # Read target R/S from isotopes
        targets = {}
        for atom in mol.GetAtoms():
            isotope = atom.GetIsotope()
            if isotope == STEREO_R_ISOTOPE:
                targets[atom.GetIdx()] = "R"
            elif isotope == STEREO_S_ISOTOPE:
                targets[atom.GetIdx()] = "S"

        # Strip isotopes so CIP runs on a clean molecule
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)

        # Set CW for all, then check and flip where needed
        for idx in targets:
            mol.GetAtomWithIdx(idx).SetChiralTag(
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
            )
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for idx, target_cip in targets.items():
            atom = mol.GetAtomWithIdx(idx)
            if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != target_cip:
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        return Chem.MolToSmiles(mol, canonical=False)

    # ══════════════════════════════════════════════════════════════════════════════
    # DOUBLE BOND (E/Z) STEREOCHEMISTRY
    # ══════════════════════════════════════════════════════════════════════════════

    def _generate_local_db_stereochem_syntax_for_mol(self, smiles: str) -> str:
        """
        Generate local double-bond stereochemistry syntax for a molecule.

        For each stereo double bond, the E/Z configuration is read from RDKit, encoded
        as an isotope label on the begin atom of the bond, and the / \\ bond directional
        symbols are stripped so that the output SMILES encodes geometry only through
        isotope markers.

        Args:
            smiles (str): Input SMILES string.

        Returns:
            str: SMILES string with E/Z encoded as isotope labels and directional
                bond symbols removed.
        """
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Collect absolute E/Z via rdCIPLabeler (sets _CIPCode = "E"/"Z" on bonds).
        # Prefer marking the begin atom; fall back to end atom if begin already
        # carries an R/S isotope (avoids overwriting tetrahedral stereo).
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
        db_targets = {}
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0 and bond.HasProp("_CIPCode"):
                cip = bond.GetProp("_CIPCode")  # "E" or "Z"
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                mark_idx = begin_idx
                if mol.GetAtomWithIdx(begin_idx).GetIsotope() in (
                    STEREO_R_ISOTOPE,
                    STEREO_S_ISOTOPE,
                ):
                    mark_idx = end_idx
                db_targets[mark_idx] = cip

        # Mark begin atoms with isotope and strip all bond stereo
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

        for idx, cip in db_targets.items():
            isotope = STEREO_E_ISOTOPE if cip == "E" else STEREO_Z_ISOTOPE
            mol.GetAtomWithIdx(idx).SetIsotope(isotope)

        smiles = Chem.MolToSmiles(mol, canonical=False)
        # Strip any residual / or \ that MolToSmiles may emit
        smiles = smiles.replace("/", "").replace("\\", "")
        return smiles

    # ── Non-deferred E/Z ─────────────────────────────────────────────────────────

    def _convert_db_syntax_non_deferred(self, smiles: str) -> str:
        """
        Replace E/Z isotope labels with inline [STEREO_E] / [STEREO_Z] tokens.

        Args:
            smiles (str): SMILES with E/Z isotope labels from
                generate_local_db_stereochem_syntax_for_mol.

        Returns:
            str: SMILES with inline E/Z tokens.
        """
        for ele in re.findall(r"\[(.*?)\]", smiles):
            iso_match = re.match(r"^(\d+)(.*)", ele)
            if not iso_match:
                continue
            isotope = int(iso_match.group(1))
            if isotope in (STEREO_E_ISOTOPE, STEREO_Z_ISOTOPE):
                original_token = "[" + ele + "]"
                atom_part = iso_match.group(2)
                token_no_isotope = "[" + atom_part + "]"
                stereo_token = (
                    STEREO_E_TOKEN if isotope == STEREO_E_ISOTOPE else STEREO_Z_TOKEN
                )
                smiles = smiles.replace(original_token, token_no_isotope + stereo_token)
        return smiles

    def _invert_db_syntax_non_deferred(self, converted_smiles: str) -> str:
        """
        Replace inline [STEREO_E] / [STEREO_Z] tokens with E/Z isotope labels.

        Args:
            converted_smiles (str): SMILES with inline E/Z tokens.

        Returns:
            str: SMILES with E/Z isotope labels.
        """
        converted_smiles = re.sub(
            r"\[([^\]]+)\]\[STEREO_E\]",
            lambda m: "[" + str(STEREO_E_ISOTOPE) + m.group(1) + "]",
            converted_smiles,
        )
        converted_smiles = re.sub(
            r"\[([^\]]+)\]\[STEREO_Z\]",
            lambda m: "[" + str(STEREO_Z_ISOTOPE) + m.group(1) + "]",
            converted_smiles,
        )
        return converted_smiles

    def _revert_db_isotopes_to_stereo(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Shared helper that reads E/Z isotope labels from a molecule, strips them,
        and restores the corresponding BondStereo on the double bonds using a
        trial-and-flip approach with FindPotentialStereo.

        Args:
            mol (Chem.Mol): RDKit molecule with E/Z isotope labels on begin atoms.

        Returns:
            Chem.Mol: Molecule with bond stereo restored and isotopes cleared.
        """
        # Read target absolute E/Z from isotopes
        db_targets = {}
        for atom in mol.GetAtoms():
            iso = atom.GetIsotope()
            if iso == STEREO_E_ISOTOPE:
                db_targets[atom.GetIdx()] = "E"
            elif iso == STEREO_Z_ISOTOPE:
                db_targets[atom.GetIdx()] = "Z"

        # Strip E/Z isotopes
        for atom in mol.GetAtoms():
            if atom.GetIsotope() in (STEREO_E_ISOTOPE, STEREO_Z_ISOTOPE):
                atom.SetIsotope(0)

        # Set an initial stereo guess (STEREOE) on each marked double bond
        db_bonds = {}
        saved_stereo_atoms = {}
        for idx in db_targets:
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    begin_neighbors = [
                        n.GetIdx()
                        for n in mol.GetAtomWithIdx(begin_idx).GetNeighbors()
                        if n.GetIdx() != end_idx
                    ]
                    end_neighbors = [
                        n.GetIdx()
                        for n in mol.GetAtomWithIdx(end_idx).GetNeighbors()
                        if n.GetIdx() != begin_idx
                    ]
                    if begin_neighbors and end_neighbors:
                        ref_begin = begin_neighbors[0]
                        ref_end = end_neighbors[0]
                        bond.SetStereoAtoms(ref_begin, ref_end)
                        bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
                        db_bonds[idx] = bond.GetIdx()
                        saved_stereo_atoms[bond.GetIdx()] = (ref_begin, ref_end)
                    break

        # Check absolute E/Z via SMILES round-trip and flip if wrong.
        # rdCIPLabeler.AssignCIPLabels corrupts the mol's internal stereo
        # representation, so CIP must be checked on a fresh molecule obtained
        # by writing + re-parsing SMILES.  Each bond is probed individually
        # (only its stereo is set) so the single CIP-labelled bond in the
        # probe mol unambiguously corresponds to the bond under test.
        flip_needed = {}
        for idx in db_targets:
            if idx not in db_bonds:
                continue
            bond_idx = db_bonds[idx]
            ref_begin, ref_end = saved_stereo_atoms[bond_idx]
            for other_bond_idx in db_bonds.values():
                mol.GetBondWithIdx(other_bond_idx).SetStereo(
                    Chem.rdchem.BondStereo.STEREONONE
                )
            mol.GetBondWithIdx(bond_idx).SetStereoAtoms(ref_begin, ref_end)
            mol.GetBondWithIdx(bond_idx).SetStereo(Chem.rdchem.BondStereo.STEREOE)
            probe_smi = Chem.MolToSmiles(mol, canonical=True)
            probe_mol = Chem.MolFromSmiles(probe_smi)
            Chem.rdCIPLabeler.AssignCIPLabels(probe_mol)
            probe_cip = None
            for pb in probe_mol.GetBonds():
                if pb.HasProp("_CIPCode"):
                    probe_cip = pb.GetProp("_CIPCode")
                    break
            flip_needed[idx] = probe_cip != db_targets[idx]

        for idx in db_targets:
            if idx not in db_bonds:
                continue
            bond_idx = db_bonds[idx]
            ref_begin, ref_end = saved_stereo_atoms[bond_idx]
            mol.GetBondWithIdx(bond_idx).SetStereoAtoms(ref_begin, ref_end)
            if flip_needed[idx]:
                mol.GetBondWithIdx(bond_idx).SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            else:
                mol.GetBondWithIdx(bond_idx).SetStereo(Chem.rdchem.BondStereo.STEREOE)

        return mol

    def _revert_local_db_stereochem_to_global_non_deferred(
        self,
        inverted_syntax_smiles: str,
    ) -> str:
        """
        Revert a SMILES with E/Z isotope labels back to standard SMILES with / \\ notation.

        Args:
            inverted_syntax_smiles (str): SMILES with E/Z isotope labels.

        Returns:
            str: Standard SMILES with restored / \\ double-bond stereochemistry.
        """
        mol = Chem.MolFromSmiles(inverted_syntax_smiles)
        mol = self._revert_db_isotopes_to_stereo(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return Chem.MolToSmiles(mol, canonical=False)

    # ── Deferred E/Z ─────────────────────────────────────────────────────────────

    def _convert_db_syntax_deferred(self, smiles: str) -> str:
        """
        Replace E/Z isotope labels with numbered [DB_STEREO_N] placeholders and append
        a deferred assignment block after a | delimiter.

        Args:
            smiles (str): SMILES with E/Z isotope labels from
                generate_local_db_stereochem_syntax_for_mol.

        Returns:
            str: SMILES with deferred E/Z syntax.
        """
        assignments = []
        counter = [0]

        def _replace_isotope(match):
            isotope = int(match.group(1))
            if isotope not in (STEREO_E_ISOTOPE, STEREO_Z_ISOTOPE):
                return match.group(0)
            counter[0] += 1
            atom_part = match.group(2)
            stereo_token = (
                STEREO_E_TOKEN if isotope == STEREO_E_ISOTOPE else STEREO_Z_TOKEN
            )
            assignments.append((counter[0], stereo_token))
            return f"[{atom_part}][DB_STEREO_{counter[0]}]"

        smiles = re.sub(r"\[(\d+)(.*?)\]", _replace_isotope, smiles)

        if assignments:
            assignment_block = "|" + "".join(
                f"[DB_STEREO_{num}]{token}" for num, token in assignments
            )
            smiles += assignment_block

        return smiles

    def _invert_db_syntax_deferred(self, converted_smiles: str) -> str:
        """
        Invert the deferred double-bond (E/Z) syntax back to isotope labels.

        Only consumes [DB_STEREO_N][STEREO_E/Z] assignments from the | block.
        Any other deferred assignments (e.g. [STEREO_CENTER_N][STEREO_R/S]) are preserved.

        Args:
            converted_smiles (str): SMILES with deferred E/Z syntax.

        Returns:
            str: SMILES with E/Z isotope labels and any remaining | block intact.
        """
        if "|" not in converted_smiles:
            return converted_smiles

        graph_part, assignment_block = converted_smiles.split("|", 1)

        # Parse DB_STEREO assignment pairs
        assignments = {}
        for match in re.finditer(
            r"\[DB_STEREO_(\d+)\]\[(STEREO_[EZ])\]", assignment_block
        ):
            center_num = int(match.group(1))
            config = match.group(2)
            isotope = STEREO_E_ISOTOPE if config == "STEREO_E" else STEREO_Z_ISOTOPE
            assignments[center_num] = isotope

        # Remove consumed assignments from block
        remaining = re.sub(r"\[DB_STEREO_\d+\]\[STEREO_[EZ]\]", "", assignment_block)

        # Replace placeholders in graph part with isotopes
        for center_num, isotope in assignments.items():
            pattern = r"\[([^\]]+)\]\[DB_STEREO_" + str(center_num) + r"\]"

            def _replace_db(match: re.Match[str], iso: int = isotope) -> str:
                return "[" + str(iso) + match.group(1) + "]"

            graph_part = re.sub(pattern, _replace_db, graph_part)

        # Reconstruct with any remaining assignment content
        remaining_parts = [p for p in remaining.split("|") if p.strip()]
        if remaining_parts:
            return graph_part + "|" + "|".join(remaining_parts)
        return graph_part

    def _revert_local_db_stereochem_to_global_deferred(
        self,
        inverted_syntax_smiles: str,
    ) -> str:
        """
        Revert a SMILES with E/Z isotope labels (from deferred inversion) back to
        standard SMILES with / \\ notation.

        Args:
            inverted_syntax_smiles (str): SMILES with E/Z isotope labels.

        Returns:
            str: Standard SMILES with restored / \\ double-bond stereochemistry.
        """
        return self._revert_local_db_stereochem_to_global_non_deferred(
            inverted_syntax_smiles
        )
