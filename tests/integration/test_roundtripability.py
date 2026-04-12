import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rdkit import Chem

from crisp_smiles.main import CRISPSmiles

CRISPSmilesConverter = CRISPSmiles()

smis = []
with open(
    "/home/csnbritt/projects/denovochem_projects/stereochem_smiles/data/zinc250k.txt",
    "r",
) as file:
    for line in file:
        smis.append(line.strip())

for deferred_val in [True, False]:
    for smiles_mode in ["canonical", "random"]:
        print(f"Testing deferred={deferred_val}, smiles_mode={smiles_mode}")
        identical_can_rand_count = 0
        failed_roundtrip_count = 0
        suffix_mismatch_count = 0
        for i, smi in enumerate(smis):
            ## Test that random/canonical and deferred/non-deferred roundtrips are consistent
            can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
            crisp_smiles_can = CRISPSmilesConverter.encode(
                can_smi, deferred=deferred_val, smiles_mode="canonical"
            )
            crisp_smiles = CRISPSmilesConverter.encode(
                can_smi, deferred=deferred_val, smiles_mode=smiles_mode
            )
            # Test that random smiles_mode produces different results from canonical most of the time
            if smiles_mode == "random" and crisp_smiles_can == crisp_smiles:
                identical_can_rand_count += 1
            restored = CRISPSmilesConverter.decode(crisp_smiles, deferred=deferred_val)
            can_restored = Chem.MolToSmiles(
                Chem.MolFromSmiles(restored), canonical=True
            )
            match = "PASS" if can_restored == can_smi else "FAIL"
            if match != "PASS":
                print(f"Failed roundtrip: {smi}")
                print(f"  Original: {can_smi}")
                print(f"  Restored: {can_restored}")
                print(f"  CRISP SMILES: {crisp_smiles}")
                failed_roundtrip_count += 1

            ## test that suffix is consistent across randomizations
            if "|" in crisp_smiles_can:
                can_crisp_suffix = crisp_smiles_can.split("|")[-1]
                can_crisp_suffix_tokens = set(
                    re.findall(r"\[([^\]]*)\]", can_crisp_suffix)
                )
                for j in range(5):
                    ran_crisp_smiles = CRISPSmilesConverter.encode(
                        can_smi, smiles_mode="random"
                    )
                    ran_crisp_suffix = ran_crisp_smiles.split("|")[-1]
                    ran_crisp_suffix_tokens = set(
                        re.findall(r"\[([^\]]*)\]", ran_crisp_suffix)
                    )
                    if can_crisp_suffix_tokens != ran_crisp_suffix_tokens:
                        suffix_mismatch_count += 1
            elif (
                "[STEREO_E]" in crisp_smiles_can
                or "[STEREO_Z]" in crisp_smiles_can
                or "[STEREO_R]" in crisp_smiles_can
                or "[STEREO_S]" in crisp_smiles_can
            ):
                can_stereo_e_count = crisp_smiles_can.count("[STEREO_E]")
                can_stereo_z_count = crisp_smiles_can.count("[STEREO_Z]")
                can_stereo_r_count = crisp_smiles_can.count("[STEREO_R]")
                can_stereo_s_count = crisp_smiles_can.count("[STEREO_S]")

                for j in range(5):
                    ran_crisp_smiles = CRISPSmilesConverter.encode(
                        can_smi, smiles_mode="random"
                    )
                    ran_stereo_e_count = ran_crisp_smiles.count("[STEREO_E]")
                    ran_stereo_z_count = ran_crisp_smiles.count("[STEREO_Z]")
                    ran_stereo_r_count = ran_crisp_smiles.count("[STEREO_R]")
                    ran_stereo_s_count = ran_crisp_smiles.count("[STEREO_S]")

                    if can_stereo_e_count != ran_stereo_e_count:
                        suffix_mismatch_count += 1
                    if can_stereo_z_count != ran_stereo_z_count:
                        suffix_mismatch_count += 1
                    if can_stereo_r_count != ran_stereo_r_count:
                        suffix_mismatch_count += 1
                    if can_stereo_s_count != ran_stereo_s_count:
                        suffix_mismatch_count += 1

        print("### REPORT ###")
        print(f"Total SMILES tested: {i + 1}")
        print(f"Failed roundtrip count: {failed_roundtrip_count}")
        if smiles_mode == "random":
            print(
                f"Identical canonical and random pct: {identical_can_rand_count / (i + 1) * 100:.2f}%"
            )
        print(f"Failed roundtrip pct: {failed_roundtrip_count / (i + 1) * 100:.2f}%")
        print(f"Suffix mismatch count: {suffix_mismatch_count}")
        print("##############")
