import scipy.stats as stats

# To understand what is this data, see the comments below
#js tensorflow
series_a = [2.65, 2.63, 2.72, 2.69, 2.50, 2.60, 2.68, 2.58, 2.59, 2.57, 2.70, 2.85, 2.69, 2.69, 2.63, 2.61, 2.63, 2.59, 2.61, 2.62, 2.54, 2.66, 2.60, 2.35, 2.68, 2.71, 2.63, 2.59, 2.54, 2.54, 2.41, 2.47, 2.57, 2.64, 2.75, 2.56, 2.64, 2.52, 2.64, 2.53, 2.69, 2.58, 2.59, 2.69, 2.54, 2.68, 2.67, 2.78, 2.53, 2.64, 2.54, 2.63, 2.50, 2.61, 2.53, 2.59, 2.56, 2.64, 2.45, 2.49, 2.50, 2.68, 2.70, 2.50, 2.59, 2.63, 2.68, 2.62, 2.49, 2.61, 2.78, 2.69, 2.59, 2.62, 2.57, 2.53, 2.53, 2.77, 2.64, 2.67, 2.54, 2.71, 2.56, 2.76, 2.62, 2.62, 2.76, 2.63, 2.56, 2.52, 2.58, 2.71, 2.74, 2.47, 2.48, 2.57, 2.68, 2.75, 2.64, 2.68]
#python pytorch
series_b = [2.68, 2.63, 2.64, 2.69, 2.60, 2.58, 2.61, 2.63, 2.63, 2.73, 2.69, 2.49, 2.74, 2.66, 2.47, 2.65, 2.59, 2.49, 2.55, 2.75, 2.71, 2.69, 2.59, 2.59, 2.65, 2.46, 2.54, 2.60, 2.59, 2.44, 2.69, 2.68, 2.72, 2.68, 2.65, 2.57, 2.59, 2.51, 2.59, 2.54, 2.59, 2.68, 2.56, 2.50, 2.64, 2.73, 2.52, 2.59, 2.67, 2.60, 2.50, 2.68, 2.53, 2.53, 2.60, 2.52, 2.63, 2.62, 2.65, 2.55, 2.50, 2.61, 2.52, 2.67, 2.55, 2.44, 2.61, 2.62, 2.53, 2.58, 2.54, 2.69, 2.62, 2.61, 2.63, 2.55, 2.58, 2.68, 2.75, 2.71, 2.59, 2.80, 2.73, 2.44, 2.52, 2.58, 2.60, 2.66, 2.86, 2.54, 2.59, 2.74, 2.71, 2.74, 2.68, 2.52, 2.86, 2.73, 2.48, 2.54]


# Perform an independent t-test
t_statistic, p_value = stats.ttest_ind(series_a, series_b)

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