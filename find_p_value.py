import scipy.stats as stats

# To understand what is this data, see the comments below
values_q_table = [2.55, 2.58, 2.52, 2.6, 2.53, 2.46, 2.64, 2.59, 2.56, 2.54, 2.6, 2.46, 2.46, 2.48, 2.52, 2.52, 2.47, 2.58, 2.53, 2.48, 2.65, 2.42, 2.53, 2.58, 2.52, 2.54, 2.52, 2.68, 2.52, 2.59, 2.55, 2.52, 2.48, 2.57, 2.62, 2.53, 2.54, 2.53, 2.53, 2.65, 2.45, 2.57, 2.46, 2.62, 2.5, 2.51, 2.56, 2.6, 2.5, 2.57, 2.71, 2.52, 2.53, 2.58, 2.48, 2.55, 2.53, 2.52, 2.58, 2.48, 2.5, 2.59, 2.48, 2.42, 2.52, 2.58, 2.39, 2.54, 2.51, 2.51, 2.49, 2.6, 2.51, 2.61, 2.49, 2.55, 2.53, 2.5, 2.54, 2.49, 2.51, 2.58, 2.64, 2.58, 2.56, 2.55, 2.58, 2.57, 2.43, 2.61, 2.63, 2.58, 2.51, 2.55, 2.45, 2.6, 2.51, 2.61, 2.45, 2.56]
values_dqn = [2.66, 2.68, 2.49, 2.67, 2.67, 2.58, 2.58, 2.53, 2.66, 2.6, 2.59, 2.62, 2.68, 2.63, 2.62, 2.64, 2.56, 2.65, 2.58, 2.66, 2.65, 2.6, 2.71, 2.53, 2.6, 2.73, 2.66, 2.65, 2.61, 2.59, 2.66, 2.66, 2.68, 2.72, 2.67, 2.68, 2.68, 2.71, 2.62, 2.73, 2.65, 2.79, 2.6, 2.7, 2.71, 2.71, 2.57, 2.62, 2.77, 2.68, 2.65, 2.62, 2.55, 2.6, 2.55, 2.66, 2.75, 2.64, 2.62, 2.65, 2.57, 2.53, 2.75, 2.6, 2.67, 2.66, 2.6, 2.59, 2.71, 2.6, 2.6, 2.62, 2.58, 2.66, 2.7, 2.63, 2.76, 2.49, 2.71, 2.63, 2.58, 2.59, 2.61, 2.67, 2.59, 2.67, 2.59, 2.6, 2.72, 2.64, 2.64, 2.64, 2.58, 2.62, 2.55, 2.73, 2.6, 2.48, 2.58, 2.58]

# Perform an independent t-test
t_statistic, p_value = stats.ttest_ind(values_q_table, values_dqn)

# Check the p-value
alpha = 0.05
if p_value < alpha:
    print("Statistically significant difference (p-value:", p_value, ")")
else:
    print("No statistically significant difference (p-value:", p_value, ")")


# The two arrays of data were extracted by runnning 100 time the same q_table arena 
# and 100 times the same DQN arena with the same strategies to compare the avg results
# over 100 tournaments and verify if there is a statistically significant difference
# with this python script.
#
# You need to clone both repositories, and run this same bash script below in both folders.

    
# #!/bin/bash

# # Get the current date and time for the filename
# current_date_time=$(date +"%Y%m%d_%H%M")

# # Filename
# filename="${current_date_time}.out"

# # Initialize an empty result string
# result=""

# # Loop to run the command 100 times
# for i in {1..100}
# do
#     echo "Running iteration $i of 100..."
#     output=$(/opt/homebrew/bin/python main.py -a grudger_recovery tit_for_tat_trustful win_stay_lose_shift always_cooperate always_defect tit_for_tat_suspicious alternate_coop random_70_cooperation | \
#     grep rl_strategy | grep Points | \
#     awk -F', ' '{for(i=1; i<=NF; i++) if ($i ~ /Avg Points\/Game = /) { match($i, /Avg Points\/Game = [0-9.]+/); print substr($i, RSTART + 18, RLENGTH - 18); }}' )

#     # Append the output to the result string with a comma
#     result="$result, $output"
# done

# # Remove the last ", " from the accumulated result
# result="${result:2}"

# # Print the accumulated result (excluding the leading comma) on the screen
# echo "Accumulated Result:"
# echo "$result"

# # Save the accumulated result to a file
# echo "$result" > "$filename"

# # Print a note on the screen
# echo "The accumulated result has been saved to $filename ."