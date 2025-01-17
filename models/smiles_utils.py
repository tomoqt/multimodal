from rdkit import Chem
import selfies as sf

def canonicalize_smiles(smiles):
    """Canonicalize SMILES string using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        return None

def safe_selfies_conversion(smiles):
    """Safely convert SMILES to SELFIES with proper error handling"""
    try:
        # First canonicalize and clean the SMILES
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return None
        # Convert to SELFIES
        return sf.encoder(canonical)
    except:
        return None

def process_smiles_to_selfies(smiles, idx=None):
    """
    Standardized helper function to convert SMILES to SELFIES with proper cleanup.
    Used across all modules for consistency.
    """
    try:
        # 1) Remove spaces
        smiles = smiles.replace(" ", "")

        # 2) Parse as SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: invalid SMILES{' at index '+str(idx) if idx is not None else ''}: {smiles}")
            return None

        # 3) Remove stereochemistry
        Chem.rdmolops.RemoveStereochemistry(mol)
        
        # 4) Create clean SMILES (canonical, no stereo)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        
        # 5) Convert to SELFIES
        return sf.encoder(canonical_smiles)
    except Exception as e:
        print(f"Error processing sequence{' at index '+str(idx) if idx is not None else ''}: {smiles}")
        print(f"Error: {e}")
        return None 