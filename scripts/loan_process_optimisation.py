#!/usr/bin/env python
# coding: utf-8

# # Loan Process Optimisation Project
# 
# This notebook covers Phase 1 (Data Hardening), Phase 2 (Sanitization & Bottleneck Analysis), Phase 3 (Transition Analysis), Phase 4 (To-Be Process), and Phase 5 (Load Balancing).

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define file paths
# Assuming script is run from src/ or we navigate relative to this file
import sys

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "visualizations")

CLEANED_FILE = os.path.join(DATA_DIR, "bpi_2017_cleaned.csv")
HARDENED_FILE = os.path.join(DATA_DIR, "bpi_2017_hardened.csv")

# Ensure images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)


# # Phase 1: Data Hardening
# 
# **Goal**: Fill missing values for case-level and offer-level attributes to create a "hardened" dataset.
# **Output**: `bpi_2017_hardened.csv`

# In[ ]:


print(f"--- Starting Phase 1 ---")
print(f"Loading raw data from {CLEANED_FILE}...")
df_phase1 = pd.read_csv(CLEANED_FILE)

case_attributes = ['CreditScore', 'case:RequestedAmount', 'MonthlyCost']
case_id_col = 'case:concept:name'

print("Filling case-level attributes (CreditScore, RequestedAmount, MonthlyCost)...")
for col in case_attributes:
    df_phase1[col] = df_phase1.groupby(case_id_col)[col].transform(lambda x: x.ffill().bfill())

offer_attributes = ['OfferedAmount']
offer_id_col = 'OfferID'

print("Filling offer-level attributes (OfferedAmount)...")
# Use dropna=False to ensure we process groups even if grouping keys have NaNs (though we care about OfferID)
for col in offer_attributes:
    df_phase1[col] = df_phase1.groupby([case_id_col, offer_id_col], dropna=False)[col].transform(lambda x: x.ffill().bfill())

print(f"Saving hardened data to {HARDENED_FILE}...")
df_phase1.to_csv(HARDENED_FILE, index=False)
print("Phase 1 complete. Hardened file saved.")


# # Phase 2: Sanitization and Bottleneck Analysis
# 
# **Goal**: Sanitize data (fix missing offers, mixed credit scores) and calculate process metrics.
# **Input**: `bpi_2017_hardened.csv`

# In[ ]:


print(f"\n--- Starting Phase 2 ---")
print(f"Loading hardened data from {HARDENED_FILE}...")
df = pd.read_csv(HARDENED_FILE)

# Convert timestamp to datetime
if 'time:timestamp' in df.columns:
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed', utc=True)

print(f"Loaded {len(df)} rows for analysis.")


# ## Step 2.1: Data Sanitization
# 
# - Fix missing `OfferedAmount` by unifying `EventID` (from O_Create Offer) and `OfferID`.
# - Create `is_viable_offer` flag.
# - Handle mixed `CreditScore` (Initial vs Final).

# In[ ]:


# 1. Handle Missing OfferedAmount
print("Unifying Offer IDs and filling OfferedAmount...")
df['TempOfferID'] = df['OfferID']
# For O_Create Offer, the OfferID is actually in the EventID column
mask_create_offer = df['concept:name'] == 'O_Create Offer'
df.loc[mask_create_offer, 'TempOfferID'] = df.loc[mask_create_offer, 'EventID']

# Now fill OfferedAmount by grouping by TempOfferID
df['OfferedAmount'] = df.groupby('TempOfferID')['OfferedAmount'].transform(lambda x: x.ffill().bfill())

# Create is_viable_offer
# True if TempOfferID is present AND OfferedAmount is populated. 
df['is_viable_offer'] = (df['TempOfferID'].notna() & df['OfferedAmount'].notna())
print(f"Created 'is_viable_offer'. Count of viable offers: {df['is_viable_offer'].sum()}")

# 2. Handle Mixed Credit Scores
print("Handling mixed CreditScores...")
df.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)

credit_score_groups = df.groupby('case:concept:name')['CreditScore']
df['initial_credit_score'] = credit_score_groups.transform('first')
df['final_credit_score'] = credit_score_groups.transform('last')

print("CreditScore columns created (initial_credit_score, final_credit_score).")


# ## Step 2.2: The Bottleneck Engine
# 
# - Calculate `Time-to-Offer`.
# - Calculate `Process Efficiency Ratio`.
# - Generate `Cycle Time Histogram` (14-Day Validation).
# - Generate `Wait Time Heatmap` (Queue Time).

# In[ ]:


# 1. Time-to-Offer
print("Calculating Time-to-Offer...")
case_starts = df[df['concept:name'] == 'A_Create Application'].groupby('case:concept:name')['time:timestamp'].min()
first_offers = df[df['concept:name'] == 'O_Create Offer'].groupby('case:concept:name')['time:timestamp'].min()

time_to_offer = (first_offers - case_starts).dt.total_seconds() / 3600.0 # Hours

print("Time-to-Offer (Hours) Stats:")
print(time_to_offer.describe())

# 2. Process Efficiency Ratio
print("Calculating Process Efficiency Ratio...")
# Queue Time Heuristic: Match 'schedule' to subsequent 'start' by Activity
transitions_df = df[df['lifecycle:transition'].isin(['schedule', 'start'])].copy()
transitions_df.sort_values(['case:concept:name', 'concept:name', 'time:timestamp'], inplace=True)

# Shift logic for queue time
transitions_df['next_transition'] = transitions_df['lifecycle:transition'].shift(-1)
transitions_df['next_timestamp'] = transitions_df['time:timestamp'].shift(-1)
transitions_df['next_case'] = transitions_df['case:concept:name'].shift(-1)
transitions_df['next_activity'] = transitions_df['concept:name'].shift(-1)
transitions_df['next_resource'] = transitions_df['org:resource'].shift(-1) 

valid_pair = (
    (transitions_df['lifecycle:transition'] == 'schedule') & 
    (transitions_df['next_transition'] == 'start') & 
    (transitions_df['case:concept:name'] == transitions_df['next_case']) & 
    (transitions_df['concept:name'] == transitions_df['next_activity'])
)

transitions_df.loc[valid_pair, 'queue_time'] = (transitions_df['next_timestamp'] - transitions_df['time:timestamp']).dt.total_seconds()
transitions_df.loc[valid_pair, 'resource_picker'] = transitions_df['next_resource']

# Active time: start -> complete
active_df = df[df['lifecycle:transition'].isin(['start', 'complete'])].copy()
active_df.sort_values(['case:concept:name', 'concept:name', 'time:timestamp'], inplace=True)

active_df['next_transition'] = active_df['lifecycle:transition'].shift(-1)
active_df['next_timestamp'] = active_df['time:timestamp'].shift(-1)
active_df['next_case'] = active_df['case:concept:name'].shift(-1)
active_df['next_activity'] = active_df['concept:name'].shift(-1)

valid_active_pair = (
    (active_df['lifecycle:transition'] == 'start') & 
    (active_df['next_transition'] == 'complete') & 
    (active_df['case:concept:name'] == active_df['next_case']) & 
    (active_df['concept:name'] == active_df['next_activity'])
)

active_df.loc[valid_active_pair, 'active_duration'] = (active_df['next_timestamp'] - active_df['time:timestamp']).dt.total_seconds()
total_active_time = active_df.groupby('case:concept:name')['active_duration'].sum()

case_end_times = df.groupby('case:concept:name')['time:timestamp'].max()
case_start_times = df.groupby('case:concept:name')['time:timestamp'].min()
total_cycle_time = (case_end_times - case_start_times).dt.total_seconds()

efficiency_ratio = total_active_time / total_cycle_time
print("Process Efficiency Ratio dataframe head:")
print(efficiency_ratio.head())

# 3. 14-Day Validation (Cycle Time Histogram)
print("Generating Cycle Time Histogram (14-Day Validation)...")
cycle_time_days = total_cycle_time / (24 * 3600)
plt.figure(figsize=(10, 6))
plt.hist(cycle_time_days, bins=50, edgecolor='k')
plt.title('Distribution of Case Cycle Times (Days)')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.axvline(14, color='r', linestyle='--', label='14 Days')
plt.legend()
plt.savefig(os.path.join(IMAGES_DIR, 'cycle_time_histogram.png'))
print("Histogram saved to cycle_time_histogram.png")

# 4. Wait Time Heatmap (Queue Time)
print("Calculating Queue Time Heatmap...")
queue_data = transitions_df[valid_pair][['queue_time', 'resource_picker']]
if not queue_data.empty:
    avg_queue_time = queue_data.groupby('resource_picker')['queue_time'].mean().sort_values(ascending=False)
    print("Top 5 Users with Highest Queue Time (Seconds):")
    print(avg_queue_time.head())
    
    plt.figure(figsize=(10, 8))
    top_50 = avg_queue_time.head(50)
    plt.barh(top_50.index.astype(str), top_50.values)
    plt.gca().invert_yaxis()
    plt.title('Average Queue Time by User (Top 50)')
    plt.xlabel('Seconds')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'wait_time_heatmap.png'))
    print("Heatmap saved to wait_time_heatmap.png")
else:
    print("No valid queue time pairs found.")


# # Phase 3: Transition Analysis & Prediction
# 
# ## Step 3.1: State Transition Analysis
# - Identify longest idle times between transitions.
# - Segment transitions by `LoanGoal`.
# - Analyze Ping-Pong effect (W_Validate application).

# In[ ]:


print(f"\n--- Starting Phase 3 --- ")

# 1. Calculate Idle Time for Transitions
print("Calculating State Transitions and Idle Times...")
df.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)

# Shift to get next activity and time
df['next_activity'] = df.groupby('case:concept:name')['concept:name'].shift(-1)
df['next_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(-1)

# Filter out last events (where next_activity is NaN)
transitions = df.dropna(subset=['next_activity']).copy()

# Calculate duration to next event (proxy for idle/transition time)
transitions['transition_duration'] = (transitions['next_timestamp'] - transitions['time:timestamp']).dt.total_seconds()
transitions['transition_pair'] = transitions['concept:name'] + " -> " + transitions['next_activity']

# Top 10 bottleneck transitions
avg_transition_time = transitions.groupby('transition_pair')['transition_duration'].mean().sort_values(ascending=False)
print("Top 10 Slowest Transition Pairs (Avg Seconds):")
print(avg_transition_time.head(10))

# 2. Segment by LoanGoal
if 'case:LoanGoal' in df.columns:
    print("\nSegmenting Bootlenecks by LoanGoal...")
    # Get top 1 bottleneck
    top_bottleneck = avg_transition_time.index[0]
    print(f"Deep Dive into Top Bottleneck: {top_bottleneck}")
    
    subset = transitions[transitions['transition_pair'] == top_bottleneck]
    goal_stats = subset.groupby('case:LoanGoal')['transition_duration'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
    print(goal_stats)
else:
    print("LoanGoal column not found.")

# 3. Ping-Pong Analysis (W_Validate application)
print("\nCalculating Ping-Pong Counts for 'W_Validate application'...")
validate_counts = df[df['concept:name'] == 'W_Validate application'].groupby('case:concept:name').size()
validate_counts.name = 'validation_count'

# Prepare Cycle Time data again for merge (using Days)
cycle_time_df = pd.DataFrame(cycle_time_days)
cycle_time_df.columns = ['cycle_time_days']

# Merge
pingpong_df = cycle_time_df.join(validate_counts, how='left').fillna(0)

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(pingpong_df['validation_count'], pingpong_df['cycle_time_days'], alpha=0.5)
plt.title('Ping-Pong Effect: Validation Count vs Cycle Time')
plt.xlabel('Number of "W_Validate application" Events')
plt.ylabel('Total Cycle Time (Days)')
plt.savefig(os.path.join(IMAGES_DIR, 'validation_pingpong.png'))
print("Scatter plot saved to validation_pingpong.png")


# ## Step 3.2: The "Happy Path" vs "Clogged Path"
# 
# - Define Group A (Fast) and Group B (Laggards).
# - Compare features (RequestedAmount, CreditScore) to find drivers.

# In[ ]:


# 1. Define Groups
print("Defining Groups A (Fast) and B (Laggards)...")
group_a = pingpong_df[pingpong_df['cycle_time_days'] <= 5].index
group_b = pingpong_df[pingpong_df['cycle_time_days'] >= 20].index

print(f"Group A (<= 5 days) count: {len(group_a)}")
print(f"Group B (>= 20 days) count: {len(group_b)}")

# 2. Feature Correlation / Comparison
# We need case-level attributes. Create a case-level dataframe.
# Taking 'first' for attributes that shouldn't change or we want initial value
case_features = df.groupby('case:concept:name').agg({
    'case:RequestedAmount': 'first',
    'initial_credit_score': 'first',
    'case:LoanGoal': 'first'
})

# Flag the groups
case_features['is_group_b'] = 0
case_features.loc[group_b, 'is_group_b'] = 1
# Filter to only A and B for comparison
analysis_df = case_features.loc[group_a.union(group_b)].copy()

print("\nComparing Features between Fast (A) vs Slow (B)...")
print("Mean Values:")
print(analysis_df.groupby('is_group_b')[['case:RequestedAmount', 'initial_credit_score']].mean())

# Correlation in the combined set
corr_amount = analysis_df['is_group_b'].corr(analysis_df['case:RequestedAmount'])
corr_score = analysis_df['is_group_b'].corr(analysis_df['initial_credit_score'])

print(f"\nCorrelation with being in Group B (Slow):")
print(f"RequestedAmount: {corr_amount:.4f}")
print(f"CreditScore: {corr_score:.4f}")

if 'case:LoanGoal' in analysis_df.columns:
    print("\nLoanGoal Distribution in Group B (Slow):")
    print(analysis_df[analysis_df['is_group_b'] == 1]['case:LoanGoal'].value_counts(normalize=True).head())
    print("\nLoanGoal Distribution in Group A (Fast):")
    print(analysis_df[analysis_df['is_group_b'] == 0]['case:LoanGoal'].value_counts(normalize=True).head())


# # Phase 4: The Strategic "To-Be" Process Map
# 
# ## Step 4.1: BPMN 2.0 Logic for AI-OCR Document Validator
# 
# **Objective**: Replace the manual "Incomplete File" loop with an automated triage system.
# 
# **To-Be Flow Idea**:
# 1. **Trigger**: `A_Submitted` event occurs (Customer submits document).
# 2. **Action**: AI-OCR Service (e.g., Google Document AI) instantly scans documents.
# 3. **Decision Gateway**: 
#     - **If Incomplete/Missing**: System triggers `Send_Automated_SMS_Email` -> Loop back to customer. (Bypasses human queue).
#     - **If Complete**: System routes case to `W_Validate application` queue.
# 
# ## Step 4.2: Theoretical Impact Calculation
# **Assumption**: We can reduce the delay of `W_Call incomplete files` -> `W_Personal Loan collection` by **90%** via automation.

# In[ ]:


print(f"\n--- Starting Phase 4: To-Be Process & Impact ---")

# 1. Calculate Current Cost of Incomplete Files
target_transition = "W_Call incomplete files -> W_Personal Loan collection"

if 'transition_duration' in transitions.columns:
    incomplete_file_transition = transitions[transitions['transition_pair'] == target_transition]
    total_delay_seconds = incomplete_file_transition['transition_duration'].sum()
    avg_delay_seconds = incomplete_file_transition['transition_duration'].mean()
    count_occurrences = len(incomplete_file_transition)
    
    print(f"Current State Analysis for '{target_transition}':")
    print(f"  - Occurrences: {count_occurrences}")
    print(f"  - Total Delay Time: {total_delay_seconds:,.0f} seconds ({total_delay_seconds/3600:,.0f} hours)")
    print(f"  - Average Delay per Case: {avg_delay_seconds:,.0f} seconds ({avg_delay_seconds/3600:.2f} hours)")
    
    # 2. Calculate Theoretical Savings (90% Reduction)
    theoretical_reduction_factor = 0.90
    time_saved_seconds = total_delay_seconds * theoretical_reduction_factor
    
    print(f"\nTheoretical 'To-Be' Impact (90% Reduction via AI-OCR):")
    print(f"  - Potential Time Saved: {time_saved_seconds:,.0f} seconds")
    print(f"  - Potential Time Saved (Hours): {time_saved_seconds/3600:,.0f} hours")
    
else:
    print("Transition data not available from previous steps. Please re-run Phase 3 logic.")


# # Phase 5: Resource Load Balancing Algorithm
# 
# ## Step 5.1: Identify Laggard Users
# Using the queue time data from Phase 2, identify the top 10% users with the highest average queue times.
# 
# ## Step 5.2: Dynamic Resource Allocator Simulation
# Redistribute tasks from 'Laggards' to 'Efficient Users' (Bottom 50% queue time). Calculate the new theoretical total queue time.

# In[ ]:


print(f"\n--- Starting Phase 5: Resource Load Balancing ---")

# 1. Identify Laggards (Top 10%)
# We use avg_queue_time from Phase 2
if 'avg_queue_time' in locals() and not avg_queue_time.empty:
    n_users = len(avg_queue_time)
    n_laggards = int(np.ceil(n_users * 0.10)) # Top 10%
    
    laggards = avg_queue_time.head(n_laggards).index
    efficient_users = avg_queue_time.tail(int(n_users * 0.50)).index # Bottom 50% (Fastest) aka lowest queue time
    
    print(f"Total Users with Queue Data: {n_users}")
    print(f"Identified {len(laggards)} Laggard Users (Top 10%).")
    print(f"Identified {len(efficient_users)} Efficient Users (Bottom 50%).")
    
    avg_efficient_queue_time = avg_queue_time[efficient_users].mean()
    print(f"Average Queue Time of Efficient Users: {avg_efficient_queue_time:.2f} seconds")
    
    # 2. Python Simulation
    print("Running Dynamic Resource Allocator Simulation...")
    
    # Work on the subset of transitions that have queue times
    simulation_df = transitions_df[transitions_df['queue_time'].notna()].copy()
    
    # Calculate As-Is Total
    total_queue_as_is = simulation_df['queue_time'].sum()
    
    # Identify tasks assigned to laggards
    laggard_tasks_mask = simulation_df['resource_picker'].isin(laggards)
    count_laggard_tasks = laggard_tasks_mask.sum()
    
    print(f"Tasks handled by Laggards: {count_laggard_tasks}")
    
    # Redistribute: Assign these tasks the average queue time of efficient users
    # (Simulating instant routing to a better resource)
    simulation_df.loc[laggard_tasks_mask, 'simulated_queue_time'] = avg_efficient_queue_time
    # Keep others as is
    simulation_df.loc[~laggard_tasks_mask, 'simulated_queue_time'] = simulation_df.loc[~laggard_tasks_mask, 'queue_time']
    
    total_queue_simulated = simulation_df['simulated_queue_time'].sum()
    
    # 3. Gap Analysis
    print(f"\nGap Analysis (As-Is vs Simulated):")
    print(f"  - Total Queue Time (As-Is): {total_queue_as_is:,.0f} seconds")
    print(f"  - Total Queue Time (Simulated): {total_queue_simulated:,.0f} seconds")
    
    improvement = total_queue_as_is - total_queue_simulated
    pct_improvement = (improvement / total_queue_as_is) * 100
    
    print(f"  - Net Improvement: {improvement:,.0f} seconds")
    print(f"  - % Reduction in Queue Time: {pct_improvement:.2f}%")
    
else:
    print("Avg Queue Time data missing. Cannot run Phase 5.")

