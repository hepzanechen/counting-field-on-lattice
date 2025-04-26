#!/bin/bash

# Get the PIDs of all your processes (excluding important ones)
top_pids=$(ps -u $USER --no-headers -o pid,comm | grep -v -E 'sshd|code|grep|systemd|\(sd-pam\)|dbus-daemon|gvfs|goa-daemon|goa-identity-se|tracker-miner-f|bash|sh|node|sleep|cpuUsage\.sh|kill_top_5_memo|ps' | awk '{print $1}')

# Check if there are any processes to kill
if [ -z "$top_pids" ]; then
    echo "No processes to kill."
    exit 0
fi

# Display the processes that are about to be killed
echo "The following processes are about to be killed:"
ps -u $USER --no-headers -o pid,comm,args | grep -v -E 'sshd|code|grep|systemd|\(sd-pam\)|dbus-daemon|gvfs|goa-daemon|goa-identity-se|tracker-miner-f|bash|sh|node|sleep|cpuUsage\.sh|kill_top_5_memo|ps'

# Ask for user confirmation
read -p "Do you want to kill these processes? (y/n): " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    # Attempt to kill the processes and handle errors
    kill -9 $top_pids
    if [ $? -eq 0 ]; then
        echo "Processes killed."
    else
        echo "Failed to kill some processes."
    fi
else
    echo "Operation cancelled."
fi