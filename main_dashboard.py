import streamlit as st
from streamlit_tree_select import tree_select
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
# Ensure these paths are correct for your system
# IMPORTANT: Replace these with your actual paths
TRADITIONAL_RESULTS_FILE = r"C:\Users\v-billreddy\Downloads\Aws_compare\traditional_model_metrics.json"
LLM_RESULTS_FILE = r"C:\Users\v-billreddy\Downloads\Aws_compare\multi_model_evaluation.json"

# --- Custom CSS for better visibility of disabled text inputs ---
st.markdown(
    """
    <style>
    /* Target disabled text inputs */
    div.stTextInput > div > div > input[disabled] {
        color: #333333; /* Darker text color (e.g., dark grey) */
        background-color: #f0f2f6; /* Slightly off-white/light grey background to differentiate */
        -webkit-text-fill-color: #333333; /* For Webkit browsers like Chrome, Safari */
        opacity: 1; /* Ensure no opacity is applied by default disabled styling */
    }

    /* Make labels for text inputs darker and bolder */
    div.stTextInput > label {
        color: #1a1a1a; /* Even darker for labels */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions for Data Loading and Structuring ---

@st.cache_data
def load_json_data(file_path):
    """Loads JSON data from a specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: Results file '{file_path}' not found. Please ensure it exists.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Check file integrity.")
        return None

@st.cache_data
def load_and_structure_traditional_data(file_path):
    """
    Loads traditional model data and structures it hierarchically:
    {algorithm: {model: {use_case_value: {"label": use_case_text, "metrics": {...}}}}}
    """
    raw_data = load_json_data(file_path)
    if not raw_data:
        return {}

    structured_data = {}
    for entry in raw_data:
        algorithm = entry.get("algorithm", "Unknown Algorithm")
        model = entry.get("model", "Unknown Model")
        use_case = entry.get("use_case", "Unknown Use Case")
        metrics = entry.get("metrics", {})

        if algorithm not in structured_data:
            structured_data[algorithm] = {}
        if model not in structured_data[algorithm]:
            structured_data[algorithm][model] = {}
        
        # Create a unique value for the use case node in the tree
        use_case_value = f"traditional_{algorithm.lower().replace(' ', '_')}_{model.lower().replace(' ', '_')}_{use_case.lower().replace(' ', '_')}"
        structured_data[algorithm][model][use_case_value] = {
            "label": use_case,
            "metrics": metrics
        }
    return structured_data

@st.cache_data
def load_and_structure_llm_data(file_path):
    """
    Loads LLM evaluation data and structures it hierarchically:
    {model_name: {prompt_value: {"label": prompt_display_text, "full_prompt": "...", "metrics": {...}}}}
    """
    raw_data = load_json_data(file_path)
    if not raw_data:
        return {}

    structured_data = {}

    for entry in raw_data:
        prompt_sno = entry.get("SNO", "Unknown SNO")
        prompt_text = entry.get("prompt", "Unknown Prompt")
        
        models_data = entry.get("models", {})
        for model_name, metrics in models_data.items():
            if model_name not in structured_data:
                structured_data[model_name] = {}
            
            prompt_value = f"llm_{model_name.lower().replace(' ', '_')}_prompt_{prompt_sno}"
            structured_data[model_name][prompt_value] = {
                "label": f"Prompt {prompt_sno}: {prompt_text[:70]}...",
                "full_prompt": prompt_text,
                "metrics": metrics
            }
    return structured_data

def create_tree_nodes(traditional_structured, llm_structured):
    """
    Constructs the tree nodes for streamlit_tree_select based on structured data.
    """
    nodes = []

    # Add Traditional Models
    for algorithm, models_data in traditional_structured.items():
        algorithm_children = []
        for model, use_cases_data in models_data.items():
            model_children = []
            for use_case_value, uc_info in use_cases_data.items():
                model_children.append({
                    "label": f"Use Case: {uc_info['label']}",
                    "value": use_case_value
                })
            algorithm_children.append({
                "label": model,
                "value": f"model_{algorithm.lower().replace(' ', '_')}_{model.lower().replace(' ', '_')}",
                "children": model_children
            })
        nodes.append({
            "label": algorithm,
            "value": f"algorithm_{algorithm.lower().replace(' ', '_')}",
            "children": algorithm_children
        })

    # Add LLM Models
    if llm_structured:
        llm_models_children = []
        for model_name, prompts_data in llm_structured.items():
            prompt_children = []
            for prompt_value, prompt_info in prompts_data.items():
                prompt_children.append({
                    "label": prompt_info["label"],
                    "value": prompt_value
                })
            llm_models_children.append({
                "label": model_name,
                "value": f"llm_model_{model_name.lower().replace(' ', '_')}",
                "children": prompt_children
            })
        nodes.append({
            "label": "LLM Prediction",
            "value": "llm_prediction_algorithms_root",
            "children": llm_models_children
        })
    else:
        nodes.append({
            "label": "LLM Prediction (No Data)",
            "value": "llm_prediction_algorithms_root_no_data",
            "children": []
        })

    return nodes

# Helper function to get the full path of parent nodes for a given leaf node
def get_path_to_node(nodes, target_value, current_path=[]):
    """Recursively finds the path (list of values) from root to target_value."""
    for node in nodes:
        new_path = current_path + [node['value']]
        if node['value'] == target_value:
            return new_path # Found the target leaf node, return its path
        if 'children' in node:
            res = get_path_to_node(node['children'], target_value, new_path)
            if res:
                return res
    return None

# --- Metric Display Function ---
def display_metrics_in_column(content_type, identifier, traditional_structured_data, llm_structured_data):
    """
    Displays the metrics in a column based on content type and identifier,
    using st.text_input with disabled=True.
    """
    st.subheader("Metric Details")
    st.markdown("---")

    metrics_to_display = {}
    
    if content_type == 'traditional_use_case':
        found_data = False
        for algo_name, models in traditional_structured_data.items():
            for model_name, use_cases in models.items():
                if identifier in use_cases:
                    use_case_info = use_cases[identifier]
                    st.write(f"**Algorithm:** {algo_name}")
                    st.write(f"**Model:** {model_name}")
                    st.write(f"**Use Case:** {use_case_info['label']}")
                    metrics_to_display = use_case_info['metrics']
                    found_data = True
                    break
            if found_data:
                break
        
        if not found_data:
            st.error("Could not find traditional use case data.")
            return

    elif content_type == 'llm_prompt':
        found_data = False
        for model_name, prompts_in_model in llm_structured_data.items():
            if identifier in prompts_in_model:
                prompt_info = prompts_in_model[identifier]
                st.write(f"**Model:** {model_name}")
                st.write(f"**Prompt:** {prompt_info['full_prompt']}")
                metrics_to_display = prompt_info['metrics']
                found_data = True
                break
        
        if not found_data:
            st.error("Could not find LLM prompt data.")
            return
    else:
        st.info("Select an item from the tree to see its metrics here.")
        return

    st.markdown("---") # Separator before metrics

    if metrics_to_display:
        for metric_name, score in metrics_to_display.items():
            st.text_input(
                label=metric_name,
                value=str(score),
                disabled=True,
                key=f"metric_input_{identifier}_{metric_name.replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '')}"
            )
    else:
        st.info("No metrics available for this selection.")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="AI Model Evaluation Hub", page_icon="🌳")

# Initialize session state variables
if 'tree_expanded_state' not in st.session_state:
    st.session_state['tree_expanded_state'] = [] # Start collapsed
if 'show_metrics_column' not in st.session_state:
    st.session_state['show_metrics_column'] = False
if 'current_metrics_type' not in st.session_state:
    st.session_state['current_metrics_type'] = None
if 'current_metrics_id' not in st.session_state:
    st.session_state['current_metrics_id'] = None
if 'selected_tree_node_value' not in st.session_state:
    st.session_state['selected_tree_node_value'] = None
# Initialize a counter for the tree key
if 'tree_rerender_key' not in st.session_state:
    st.session_state['tree_rerender_key'] = 0


st.title("🌳 AI Model Evaluation Hub")
st.markdown("Select a **Use Case** or **LLM Prompt** from the tree to view its metrics. Only one selection can be active at a time.") # Shortened instruction

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Hierarchy")
    
    traditional_structured_data = load_and_structure_traditional_data(TRADITIONAL_RESULTS_FILE)
    llm_structured_data = load_and_structure_llm_data(LLM_RESULTS_FILE)
    nodes_data = create_tree_nodes(traditional_structured_data, llm_structured_data)

    # Store the previous selected value to detect changes for rerun
    previous_selected_node_value = st.session_state['selected_tree_node_value']
    
    # Display the tree select component with a dynamic key
    selected_tree_output = tree_select(
        nodes_data,
        checked=[st.session_state['selected_tree_node_value']] if st.session_state['selected_tree_node_value'] else None,
        expanded=st.session_state['tree_expanded_state'],
        check_model='leaf', # Only leaf nodes (Use Cases/Prompts) can be checked
        key=f'main_tree_select_{st.session_state["tree_rerender_key"]}' # Use dynamic key here
    )

    # --- Logic to enforce single selection and auto-collapse ---
    
    newly_checked_from_tree = selected_tree_output.get('checked', [])
    
    # Flag to determine if we need to trigger a rerun at the end of this logic block
    should_rerun = False

    if newly_checked_from_tree:
        # Determine the user's *intended* single selection.
        current_candidate_value = None
        if len(newly_checked_from_tree) == 1:
            # If only one item is checked, that's the clear selection.
            current_candidate_value = newly_checked_from_tree[0]
        elif previous_selected_node_value in newly_checked_from_tree:
            # If previous is still checked, and there are others, the user likely clicked one of the "others"
            other_items = [item for item in newly_checked_from_tree if item != previous_selected_node_value]
            if other_items:
                current_candidate_value = other_items[0] # Take the first "new" one
            else: 
                # This rare case means previous_selected_node_value is the only one in the list,
                # but `len(newly_checked_from_tree) > 1` was somehow true. Fallback to previous.
                current_candidate_value = previous_selected_node_value
        else: 
            # Previous is not in the list, and multiple new ones are somehow checked. Take the first.
            current_candidate_value = newly_checked_from_tree[0]


        # Only update if a valid candidate was found AND it's different from current state
        if current_candidate_value and current_candidate_value != st.session_state['selected_tree_node_value']:
            st.session_state['selected_tree_node_value'] = current_candidate_value
            st.session_state['tree_rerender_key'] += 1 # Increment the key to force re-render
            should_rerun = True # A new item was selected, so rerun.

            # Determine the type and ID for the new selection (for the right dashboard)
            is_traditional_use_case = False
            for algo_data in traditional_structured_data.values():
                for model_data in algo_data.values():
                    if current_candidate_value in model_data:
                        is_traditional_use_case = True
                        break
                if is_traditional_use_case:
                    break
            
            is_llm_prompt = False
            for model_data in llm_structured_data.values():
                if current_candidate_value in model_data:
                    is_llm_prompt = True
                    break

            if is_traditional_use_case:
                st.session_state['show_metrics_column'] = True
                st.session_state['current_metrics_type'] = 'traditional_use_case'
                st.session_state['current_metrics_id'] = current_candidate_value
            elif is_llm_prompt:
                st.session_state['show_metrics_column'] = True
                st.session_state['current_metrics_type'] = 'llm_prompt'
                st.session_state['current_metrics_id'] = current_candidate_value
            else:
                # Fallback for invalid/unmatched selection
                st.session_state['show_metrics_column'] = False
                st.session_state['current_metrics_type'] = None
                st.session_state['current_metrics_id'] = None
                st.session_state['selected_tree_node_value'] = None 
                
            # --- AUTO-COLLAPSE/EXPAND LOGIC ---
            path_to_selected_node = get_path_to_node(nodes_data, current_candidate_value)
            if path_to_selected_node:
                new_expanded_state = []
                for node_val in path_to_selected_node:
                    # Only add parent nodes (algorithm, model, and top LLM root) to expanded list
                    # Exclude the leaf node itself from `expanded` list
                    if not (node_val.startswith('traditional_') and 'use_case' in node_val) and \
                       not (node_val.startswith('llm_') and 'prompt' in node_val):
                        new_expanded_state.append(node_val)
                
                # Ensure the root LLM node is expanded if any LLM child is selected
                if current_candidate_value and current_candidate_value.startswith('llm_') and 'llm_prediction_algorithms_root' not in new_expanded_state:
                    new_expanded_state.insert(0, 'llm_prediction_algorithms_root')

                st.session_state['tree_expanded_state'] = list(set(new_expanded_state))
            else: 
                 st.session_state['tree_expanded_state'] = []

    # Case: Nothing is checked from the tree component's output, and we previously had something selected.
    elif not newly_checked_from_tree and st.session_state['selected_tree_node_value'] is not None:
        st.session_state['show_metrics_column'] = False
        st.session_state['current_metrics_type'] = None
        st.session_state['current_metrics_id'] = None
        st.session_state['selected_tree_node_value'] = None # Clear current selection
        st.session_state['tree_expanded_state'] = [] # Collapse all branches
        st.session_state['tree_rerender_key'] += 1 # Increment key to force reset
        should_rerun = True

    # If any significant state change occurred, trigger a rerun
    if should_rerun:
        st.rerun()

with col2:
    if st.session_state['show_metrics_column']:
        display_metrics_in_column(
            st.session_state['current_metrics_type'],
            st.session_state['current_metrics_id'],
            traditional_structured_data,
            llm_structured_data
        )
    else:
        st.subheader("Metric Details")
        st.markdown("---")
        st.info("Select a Use Case or LLM Prompt from the left-side tree to view its metrics here.")