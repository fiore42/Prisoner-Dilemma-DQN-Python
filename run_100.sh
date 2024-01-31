#!/bin/bash

# Get the current date and time for the filename
current_date_time=$(date +"%Y%m%d_%H%M")

# Filename
filename="${current_date_time}.out"

# Initialize an empty result string
result=""

# Loop to run the command 100 times
for i in {1..100}
do
    echo "Running iteration $i of 100..."
    output=$(/opt/homebrew/bin/python main.py -a grudger_recovery tit_for_tat_trustful win_stay_lose_shift always_cooperate always_defect tit_for_tat_suspicious alternate_coop random_70_cooperation | \
    grep rl_strategy | grep Points | \
    awk -F', ' '{for(i=1; i<=NF; i++) if ($i ~ /Avg Points\/Game = /) { match($i, /Avg Points\/Game = [0-9.]+/); print substr($i, RSTART + 18, RLENGTH - 18); }}' )

    # Append the output to the result string with a comma
    result="$result, $output"
done

# Remove the last ", " from the accumulated result
result="${result:2}"

# Print the accumulated result (excluding the leading comma) on the screen
echo "Accumulated Result:"
echo "$result"

# Save the accumulated result to a file
echo "$result" > "$filename"

# Print a note on the screen
echo "The accumulated result has been saved to $filename ."