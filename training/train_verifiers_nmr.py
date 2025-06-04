import os
from datasets import load_dataset
import verifiers as vf
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def tanimoto_reward(prompt, completion, answer, parser: vf.XMLParser, **kwargs) -> float:
    """Compute Tanimoto similarity between predicted and target SMILES."""
    try:
        pred = parser.parse_answer(completion) or ""
        mol_pred = Chem.MolFromSmiles(pred)
        mol_target = Chem.MolFromSmiles(answer)
        if mol_pred is None or mol_target is None:
            return 0.0
        fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
        fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
        return float(DataStructs.TanimotoSimilarity(fp_pred, fp_target))
    except Exception:
        return 0.0


def load_nmr_dataset(path: str):
    """Load dataset of NMR spectra and SMILES.

    Expects JSON lines with keys 'nmr' and 'smiles'.
    Returns Hugging Face dataset object with 'question' and 'answer' columns.
    """
    data_files = {
        'train': os.path.join(path, 'train.jsonl'),
        'validation': os.path.join(path, 'val.jsonl')
    }
    ds = load_dataset('json', data_files=data_files)
    ds = ds.rename_columns({'nmr': 'question', 'smiles': 'answer'})
    return ds['train'], ds['validation']


def main(data_path: str, model_name: str):
    train_ds, val_ds = load_nmr_dataset(data_path)

    parser = vf.XMLParser(['think', 'answer'])
    rubric = vf.Rubric(
        funcs=[lambda prompt, completion, answer, **kw: tanimoto_reward(prompt, completion, answer, parser)],
        weights=[1.0],
        parser=parser
    )
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    system_prompt = (
        "You are a chemist. Given the following NMR spectra description, predict the corresponding molecule in SMILES format. "
        "Respond in the following format:\n" + parser.get_format_str()
    )

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=val_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=32,
    )

    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    args = vf.grpo_defaults(run_name='nmr_verifiers')
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args,
    )
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train NMR to SMILES model with verifiers GRPO')
    parser.add_argument('--data-path', type=str, required=True, help='Path with train.jsonl and val.jsonl')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    args = parser.parse_args()
    main(args.data_path, args.model_name)
