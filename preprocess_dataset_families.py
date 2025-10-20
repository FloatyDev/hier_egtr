import json
import os
import torch # Keep torch for orig2famidx tensor operations
import argparse
from collections import defaultdict
from tqdm import tqdm # Import tqdm for progress bars
from model.util import get_super_rel_map, get_orig2idx

def filter_coco_annotations(original_coco_data, relevant_image_ids_set):
    """
    Filters COCO-style annotations dictionary to keep only relevant images
    and their corresponding annotations based on the provided keys.

    Args:
        original_coco_data (dict): The loaded COCO JSON data (e.g., from train.json)
                                   containing keys like 'images', 'annotations', 'categories'.
        relevant_image_ids_set (set): A set of integer image IDs to keep.

    Returns:
        dict: A new dictionary containing only the filtered images and annotations,
              along with original info, licenses, and categories.
    """
    filtered_data = {
        'categories': original_coco_data.get('categories', []),
        'images': [],
        'annotations': []
    }

    original_image_ids_in_coco = set()

    if 'images' in original_coco_data:
        for img_info in original_coco_data['images']:
            img_id = img_info.get('id')
            if img_id is None:
                print(f"Warning: Image entry found without 'id'. Skipping: {img_info}")
                continue
            original_image_ids_in_coco.add(img_id) # Track ID from coco file
            if img_id in relevant_image_ids_set:
                filtered_data['images'].append(img_info)
    else:
        print("Warning: 'images' key not found in original COCO data.")

    kept_image_ids = {img['id'] for img in filtered_data['images']}

    # Filter annotations: Keep only those whose 'image_id' matches a kept image
    if 'annotations' in original_coco_data:
        for ann_info in original_coco_data['annotations']:
            ann_image_id = ann_info.get('image_id')
            if ann_image_id is None:
                print(f"Warning: Annotation entry found without 'image_id'. Skipping: {ann_info}")
                continue
            if ann_image_id in kept_image_ids:
                filtered_data['annotations'].append(ann_info)
    else:
        print("Warning: 'annotations' key not found in original COCO data.")

    # Check for IDs that had relations but weren't in the original COCO images list
    missing_from_coco = relevant_image_ids_set - original_image_ids_in_coco
    if missing_from_coco:
        print(f"Warning: {len(missing_from_coco)} image IDs had relations but were NOT found in the original COCO 'images' list. Examples: {list(missing_from_coco)[:5]}")

    return filtered_data

def preprocess_relations_and_annotations_by_family(data_dir):
    """
    Splits rel.json and filters train/val/test.json based on relationship family.
    Assumes original relation IDs are 1-50, maps them to 0-49 internally.

    Args:
        data_dir (str): Directory containing rel.json, train.json, val.json, test.json.
    """
    input_rel_path = os.path.join(data_dir, "rel.json")
    output_dir = data_dir # Save in the same directory

    print(f"Loading original relation annotations from: {input_rel_path}")
    try:
        with open(input_rel_path, 'r') as f:
            original_rel_data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_rel_path}: {e}")
        return

    # --- Get Mappings (Based on 50 non-background relations, indexed 0-49) ---
    print("Generating family and index mappings...")
    try:
        orig2fam = get_super_rel_map()
        orig2famidx, num_geo, num_poss, num_sem = get_orig2idx()
    except Exception as e:
         print(f"Error getting mappings from model.util: {e}")
         return
    map_length = len(orig2fam)
    # ... (Category checks remain the same) ...
    original_rel_categories = original_rel_data.get('rel_categories')
    if not original_rel_categories:
        print("Error: 'rel_categories' key not found in input relation JSON.")
        return
    num_original_cats_with_bg = len(original_rel_categories)
    has_background = num_original_cats_with_bg == map_length + 1 and original_rel_categories[0] == "__background__"

    # --- Prepare New Data Structures for Relations ---
    family_rel_data_map = {0: {'rel_categories': []}, 1: {'rel_categories': []}, 2: {'rel_categories': []}}
    family_names = {0: "geometric", 1: "possessive", 2: "semantic"}

    temp_family_cats = {0: [], 1: [], 2: []}
    start_idx_cats = 1 if has_background else 0
    for idx, cat_name in enumerate(original_rel_categories[start_idx_cats:], start=start_idx_cats):
         map_idx = idx - start_idx_cats
         if 0 <= map_idx < map_length:
             family_id = orig2fam[map_idx]
             new_idx = orig2famidx[map_idx].item()
             temp_family_cats[family_id].append((new_idx, cat_name))
    for family_id in range(3):
         sorted_cats = sorted(temp_family_cats[family_id], key=lambda x: x[0])
         family_rel_data_map[family_id]['rel_categories'] = [name for _, name in sorted_cats]
         print(f"Family '{family_names[family_id]}' has {len(sorted_cats)} relation categories.")


    splits = ['train', 'val', 'test']
    image_ids_in_rel_json = {split: set() for split in splits} # Store as int

    relevant_image_ids_per_family = {split: {0: set(), 1: set(), 2: set()} for split in splits}
    found_background_triplet = False
    max_orig_rel_idx_found = -1

    for split in splits:
        if split not in original_rel_data:
            print(f"Warning: Split '{split}' not found in input relation file. Skipping.")
            continue

        print(f"Processing split '{split}' for relations...")
        for fam_dict in family_rel_data_map.values():
            fam_dict[split] = defaultdict(list) # Use defaultdict for relations

        original_split_data = original_rel_data[split]

        for image_id_str, triplets in tqdm(original_split_data.items(), desc=f"Scanning {split} relations"):
            try:
                image_id = int(image_id_str) # Use integer IDs internally
            except ValueError:
                print(f"Warning: Could not parse image ID '{image_id_str}'. Skipping.")
                continue

            image_ids_in_rel_json[split].add(image_id) # Add int ID
            if not isinstance(triplets, list): continue

            # Track if this image has relations for each family
            image_has_family_relation = {0: False, 1: False, 2: False}

            for triplet in triplets:
                if len(triplet) != 3: continue
                s, o, r_orig = triplet
                if r_orig > max_orig_rel_idx_found: max_orig_rel_idx_found = r_orig
                if r_orig == 0:
                    found_background_triplet = True
                    continue
                map_idx = r_orig - 1
                if not (0 <= map_idx < map_length): continue

                family_id = orig2fam[map_idx]
                new_rel_idx = orig2famidx[map_idx].item()

                # Add integer image ID to the relevant set for this family
                relevant_image_ids_per_family[split][family_id].add(image_id)
                image_has_family_relation[family_id] = True # Mark that this image is relevant

                # Append filtered relation using string ID key for JSON structure
                target_dict = family_rel_data_map[family_id]
                target_dict[split][image_id_str].append([s, o, new_rel_idx])

        print(f"Split '{split}': Found relations for {len(relevant_image_ids_per_family[split][0])} geo, "
              f"{len(relevant_image_ids_per_family[split][1])} poss, "
              f"{len(relevant_image_ids_per_family[split][2])} sem images.")

    # --- Save Filtered Relation Files ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving filtered relation files to: {output_dir}")
    for family_id, data_dict in family_rel_data_map.items():
        family_name = family_names[family_id]
        output_path = os.path.join(output_dir, f"rel_{family_name}.json")
        try:
            # Convert defaultdicts back to regular dicts
            for split in splits:
                 if split in data_dict:
                      data_dict[split] = dict(data_dict[split])
            with open(output_path, 'w') as f:
                json.dump(data_dict, f) # Use indent=None for smaller files
            print(f"Successfully saved {output_path}")
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

    # --- Filter Main Annotation Files (train/val/test.json) ---
    print(f"\nFiltering main COCO annotation files (train/val/test.json)...")
    original_coco_ids_found = {split: set() for split in splits}

    for split in splits:
        original_coco_path = os.path.join(data_dir, f"{split}.json")
        print(f"Loading original COCO annotations from: {original_coco_path}")
        try:
            with open(original_coco_path, 'r') as f:
                original_coco_data = json.load(f)
            # Store the image IDs found in this COCO file
            if 'images' in original_coco_data:
                 # Ensure IDs are integers
                 original_coco_ids_found[split] = {int(img['id']) for img in original_coco_data['images']}
            else:
                 print(f"Warning: 'images' key missing in {original_coco_path}")
                 original_coco_ids_found[split] = set()
        except Exception as e:
            print(f"Error loading {original_coco_path}: {e}. Cannot filter for this split.")
            continue # Skip filtering for this split

        # Filter and save for each family
        for family_id in range(3):
            family_name = family_names[family_id]
            # Use the integer set of relevant IDs for filtering
            relevant_ids_set = relevant_image_ids_per_family[split][family_id]
            print(f"Filtering {split}.json for family '{family_name}' ({len(relevant_ids_set)} relevant images)...")

            # Call the filtering function using the keys from the user prompt
            filtered_coco_data = filter_coco_annotations(original_coco_data, relevant_ids_set)

            # Define output path for the filtered COCO file
            output_coco_path = os.path.join(output_dir, f"{split}_{family_name}.json")
            try:
                with open(output_coco_path, 'w') as f:
                    json.dump(filtered_coco_data, f, indent=2) # Added indent for readability
                print(f"Successfully saved filtered annotations to {output_coco_path}")
                print(f" -> Kept {len(filtered_coco_data['images'])} images and {len(filtered_coco_data['annotations'])} annotations.")
            except Exception as e:
                print(f"Error saving {output_coco_path}: {e}")

    # --- Final Checks (using integer sets) ---
    print("\n--- Sanity Checks ---")
    # ... (Checks remain the same as previous version) ...
    print(f"Maximum original relation index found in triplets: {max_orig_rel_idx_found}")
    if found_background_triplet: print("Warning: Triplets with relation index 0 (assumed background) were found and skipped.")
    else: print("No triplets with relation index 0 (background) were found.")

    all_assigned_ids_int = {split: set().union(*relevant_image_ids_per_family[split].values()) for split in splits}
    for split in splits:
         unassigned_ids = image_ids_in_rel_json[split] - all_assigned_ids_int[split]
         if unassigned_ids:
              print(f"Warning: {len(unassigned_ids)} image IDs in rel.json['{split}'] were NOT assigned to any family! Examples: {list(unassigned_ids)[:5]}")
         else:
              print(f"All {len(image_ids_in_rel_json[split])} image IDs from rel.json['{split}'] were assigned to at least one family.")

         coco_ids = original_coco_ids_found[split]
         rel_ids = image_ids_in_rel_json[split]
         if not coco_ids: continue

         ids_only_in_coco = coco_ids - rel_ids
         ids_only_in_rel = rel_ids - coco_ids
         if ids_only_in_coco:
              print(f"Info: {len(ids_only_in_coco)} image IDs are in {split}.json but NOT in rel.json['{split}']. Examples: {list(ids_only_in_coco)[:5]}")
         if ids_only_in_rel:
              print(f"Warning: {len(ids_only_in_rel)} image IDs are in rel.json['{split}'] but NOT in {split}.json. Examples: {list(ids_only_in_rel)[:5]}.")


    print("\nPreprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split VG annotations by relation family and filter COCO files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/visual_genome",
        help="Directory containing rel.json, train.json, val.json, test.json."
    )
    args = parser.parse_args()
    preprocess_relations_and_annotations_by_family(args.data_dir)
