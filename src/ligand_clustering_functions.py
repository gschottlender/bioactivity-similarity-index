import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans
import random

def load_properties_database(file_path):
    """Load a dataset of physicochemical properties from a CSV file."""
    return pd.read_csv(file_path, index_col=0)

def filter_ligands(properties_df, ligand_properties):
    """Filter eligible ligands according to DUD-E criteria, based on the properties of a reference ligand."""
    filtered_df = properties_df[(properties_df['mw'].between(ligand_properties['mw'] - 25, ligand_properties['mw'] + 25)) &
                                (properties_df['logP'].between(ligand_properties['logP'] - 1, ligand_properties['logP'] + 1)) &
                                (properties_df['rot_bonds'].between(ligand_properties['rot_bonds'] - 2, ligand_properties['rot_bonds'] + 2)) &
                                (properties_df['h_acceptors'].between(ligand_properties['h_acceptors'] - 1, ligand_properties['h_acceptors'] + 1)) &
                                (properties_df['h_donors'].between(ligand_properties['h_donors'] - 1, ligand_properties['h_donors'] + 1)) &
                                (properties_df['charge'] == ligand_properties['charge'])]
    return filtered_df

def get_ligand_scaffolds(smiles_dict):
    """Precompute Bemis-Murcko scaffolds for all ligands in a SMILES dictionary."""
    scaffolds = {}
    for idx, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds[idx] = scaffold_smiles
    return scaffolds

def retrieve_ligand_properties(ligand_id, properties_df):
    """Retrieve the properties of a ligand from the database."""
    if ligand_id in properties_df.index:
        return properties_df.loc[ligand_id].to_dict()
    else:
        raise ValueError(f"Ligando {ligand_id} no encontrado en la base de datos.")


def calculate_similarity_filter(ligand_id, filtered_properties, fps, threshold=0.5):
    """Calculate topologically distinct decoys within a filtered subset of ligands."""
    ligand_fp = fps[ligand_id]
    pre_decoys = list(filtered_properties.index)
    pre_decoys_fps = [fps[c] for c in pre_decoys]
    tanimoto_pre_decoys = BulkTanimotoSimilarity(ligand_fp,pre_decoys_fps)

    decoys = [pre_decoys[i] for i in range(len(pre_decoys_fps)) if tanimoto_pre_decoys[i]<threshold]
    return decoys

def bemis_murcko_clustering(decoys, scaffolds):
    """Apply Bemis–Murcko clustering to identify unique structures among the ligands."""
    scaffold_dict = {}
    for d in decoys:
        scaffold_smiles = scaffolds[d]
        if scaffold_smiles not in scaffold_dict:
            scaffold_dict[scaffold_smiles] = []
        scaffold_dict[scaffold_smiles].append(d)
    clustered_ids = [ligands[0] for ligands in scaffold_dict.values()]  # Select one representative per cluster
    return clustered_ids

def butina_clustering(compounds_list, fingerprints_dict, threshold=0.4):
    """Perform Butina clustering and select a representative compound from each cluster."""
    fingerprints = [fingerprints_dict[cmp] for cmp in compounds_list if cmp in fingerprints_dict]
    ids = [cmp for cmp in compounds_list if cmp in fingerprints_dict]

    # Salculate similarity matrix
    n_fps = len(fingerprints)
    similarity_matrix = []
    for i in range(n_fps):
        row = [1.0 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]) for j in range(i)]
        similarity_matrix.extend(row)

    # Perform Butina Clustering
    clusters = Butina.ClusterData(similarity_matrix, n_fps, threshold, isDistData=True)

    # Select one representative per cluster
    representative_ids = [ids[cluster[0]] for cluster in clusters]
    return representative_ids

def generate_decoys_from_properties(l, ligs_props, fps, scaffolds, threshold=0.4,max_decoys = 100):
    """Generate decoys for a ligand using a database of precomputed physicochemical properties."""
    random.seed(10)
    l_prop = retrieve_ligand_properties(l,ligs_props)
    pre_decoys = filter_ligands(ligs_props,l_prop)
    decoys = calculate_similarity_filter(l, pre_decoys, fps, threshold)
    final_decoys = bemis_murcko_clustering(decoys,scaffolds)
    if len(final_decoys) > max_decoys:
        final_decoys = random.sample(final_decoys,max_decoys)

    return final_decoys

def compound_k_means_clustering(l_ligs,ligs_fps,n_clusters=100):
    """Obtains a subset of representatives using K-means clustering"""
    vecs = [ligs_fps[l] for l in l_ligs]
    vecs_array = np.array(vecs).astype(float)
    kmeans = KMeans(n_clusters=n_clusters,n_init='auto', random_state=10)
    kmeans.fit(vecs_array)
    labels = kmeans.labels_
    # 'labels' is an array of cluster labels and 'l_ligs' is your list of elements
    # Initialize a dictionary to store the representatives of each cluster
    representatives = {}
    
    # Iterate over the cluster labels
    for i, label in enumerate(labels):
        label = int(label)
        # If the cluster is not yet in the dictionary, add it with the corresponding element as its representative
        if label not in representatives:
            representatives[label] = l_ligs[i]
    
    # Convert the dictionary values into a list of representatives
    ligs = list(representatives.values())

    return ligs

def get_decoys(l_pos,decoys,scaffolds,n_decoys_per_lig=10):
    """Retrieves a set of decoys for a specific target"""
    random.seed(10)
    total_decoys = []
    for l in l_pos:
        # Obtain respective decoys for each active compound and add them to the target decoy list
        decoys_l = decoys[l]
        # Reduces decoys for each active compound according to established limit
        if len(decoys_l) > n_decoys_per_lig:
            decoys_l = random.sample(decoys_l,n_decoys_per_lig)
        total_decoys+=decoys_l
    # Clusterizes final list with BM clustering to simplify redundances
    total_decoys = bemis_murcko_clustering(total_decoys,scaffolds)
    return total_decoys

def compute_tanimoto(pairs,fps):
    """Retrieves Tanimoto Similarity for compound pairs"""
    tanimotos = []
    for l1,l2 in pairs:
        fp_l1 = fps[l1]
        fp_l2 = fps[l2]
        tanimoto = DataStructs.TanimotoSimilarity(fp_l1,fp_l2)
        tanimotos.append(tanimoto)
    return tanimotos