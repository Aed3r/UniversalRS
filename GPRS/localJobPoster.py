import os
import subprocess
import time

DEFAULT_SYNC_FOLDER = "./GPRS"
DEFAULT_SCRIPT_NAME = "job_script.sh"
REMOTE_LOGS_LOC = "./logs"

# Job states, dont change this
JOB_STATES = {
    "qw": "Pending",
    "hqw": "Pending, system hold",
    "hRwq": "Pending, user and system hold, re-queue",
    "r": "Running",
    "t": "Transferring",
    "Rr": "Running, re-submit",
    "Rt": "Transferring, re-submit",
    "s": "Obsuspended",
    "ts": "Obsuspended",
    "S": "Queue suspended",
    "tS": "Queue suspended",
    "T": "Queue suspended by alarm",
    "tT": "Queue suspended by alarm",
    "Rs": "Allsuspended with re-submit",
    "Rts": "Allsuspended with re-submit",
    "RS": "Allsuspended with re-submit",
    "RtS": "Allsuspended with re-submit",
    "RT": "Allsuspended with re-submit",
    "RtT": "Allsuspended with re-submit",
    "Eqw": "Allpending states with error",
    "Ehqw": "Allpending states with error",
    "EhRqw": "Allpending states with error",
    "dr": "Deleted",
    "dt": "Deleted",
    "dRr": "Deleted",
    "dRt": "Deleted",
    "ds": "Deleted",
    "dS": "Deleted",
    "dT": "Deleted",
    "dRs": "Deleted",
    "dRS": "Deleted",
    "dRT": "Deleted"
}

# Launch job
def launchJob(script_name=DEFAULT_SCRIPT_NAME, script_folder=DEFAULT_SYNC_FOLDER):
    """Submit the job locally using qsub or equivalent."""
    print("Launching job...")
    try:
        # Check if remote logs folder exists
        if not os.path.exists(REMOTE_LOGS_LOC):
            os.mkdir(REMOTE_LOGS_LOC)
        
        script_loc = os.path.join(script_folder, script_name).replace('\\', '/')
        cmd = f"qsub -o {REMOTE_LOGS_LOC} -e {REMOTE_LOGS_LOC} -V {script_loc}"

        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            output = output.decode().strip()
            job_id = output.split()[2]  # Extract job ID from qsub output
            print(f"Job {job_id} submitted successfully!")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e.output.decode().strip()}")
            return None
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None
            
def monitor_job(job_id, job_name=DEFAULT_SCRIPT_NAME, verbose=True, failOn=[], nStdoutLines=-1):
    """Periodically check job status using qstat, retrieve and print execution results."""
    check_count = 1
    last_stdout_lines = []  # A list to track the last printed lines from stdout
    last_stderr_lines = []  # A list to track the last printed lines from stderr
    first_output_line_printed = False  # Flag to indicate when the first line of the output is printed
    stdout_location = os.path.join(REMOTE_LOGS_LOC, f"{job_name}.o{job_id}")
    stderr_location = os.path.join(REMOTE_LOGS_LOC, f"{job_name}.e{job_id}")
    lastState = None
    retries = 5

    def print_updated_lines(current_output_lines, last_lines):
        """Helper function to print the new lines that were not printed before."""
        new_lines = []
        
        # Determine the starting index for new lines
        if last_lines:
            last_line = last_lines[-1]
            try:
                start_idx = current_output_lines.index(last_line) + 1
            except ValueError:
                # This means that the last_line is no longer in the current_output_lines,
                # which is unusual but could happen. Resetting start index to 0 to be safe.
                start_idx = 0
        else:
            start_idx = 0
        
        # Extract new lines
        new_lines = current_output_lines[start_idx:]
        
        # Print new lines and update last_lines
        if new_lines:
            for line in new_lines:
                print(line, end="")
            last_lines = current_output_lines  # Update last_lines to the current output
        
        return last_lines  # Return the updated last_lines
    
    def failJobOnOutput(output_lines, failOn):
        """Helper function to fail job if output contains any of the failOn strings."""
        failCount = 0
        for line in output_lines:
            for failString, failAfter in failOn:
                if failString in line: 
                    failCount += 1

                    if failCount >= failAfter:
                        print(f"Job failed because of: {failString}")
                        print(f"Output: {line}")
                        return True
        return False

    try:
        while True:
            cmd = "qstat"
            try:
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                output_lines = output.decode().strip().splitlines()
            except subprocess.CalledProcessError as e:
                print(f"Error fetching job status: {e.output.decode().strip()}")
                exit()

            # Extract line corresponding to the job of interest
            job_line = next((line for line in output_lines if line.strip().startswith(str(job_id))), None)

            if job_line:
                state = job_line.split()[4]

                if lastState == None or lastState != state:
                    print(f"Job {job_id} is in state: {JOB_STATES.get(state, str(state))}")
                    lastState = state
                
                if state in ['r', 'Rr']:  # If the job is in a running state
                    if not os.path.exists(stdout_location) or not os.path.exists(stderr_location):
                        if retries == 0:
                            print("Output files not found! Retrying 5 times failed. Exiting...")
                            return False
                        print(f"Output files not found! Retrying {retries} more times...")
                        retries -= 1
                        time.sleep(10)
                        continue

                    with open(stdout_location, 'r') as stdout_file:
                        current_stdout_lines = stdout_file.readlines()

                    with open(stderr_location, 'r') as stderr_file:
                        current_stderr_lines = stderr_file.readlines()

                    if not first_output_line_printed and (current_stdout_lines or current_stderr_lines):
                        first_output_line_printed = True

                    if verbose:
                        # Print updated lines for stdout
                        last_stdout_lines = print_updated_lines(current_stdout_lines, last_stdout_lines)

                        # Print updated lines for stderr
                        last_stderr_lines = print_updated_lines(current_stderr_lines, last_stderr_lines)

                    if nStdoutLines > 0:
                        if len(current_stdout_lines) >= nStdoutLines:
                            print(f"Stopping monitoring job {job_id} because {nStdoutLines} lines of stdout have been printed.")
                            return True
                    
                    fail = failJobOnOutput(current_stdout_lines, failOn)
                    if fail:
                        return False
                    
                    fail = failJobOnOutput(current_stderr_lines, failOn)
                    if fail:
                        return False

            else:
                if first_output_line_printed:
                    print(f"Job {job_id} has completed!")
                else:
                    print(f"Job {job_id} has completed or does not exist!")

                with open(stdout_location, 'r') as stdout_file:
                    current_stdout_lines = stdout_file.readlines()

                with open(stderr_location, 'r') as stderr_file:
                    current_stderr_lines = stderr_file.readlines()
                
                if verbose:
                    # Print updated lines for stdout
                    last_stdout_lines = print_updated_lines(current_stdout_lines, last_stdout_lines)

                    # Print updated lines for stderr
                    last_stderr_lines = print_updated_lines(current_stderr_lines, last_stderr_lines)

                fail = failJobOnOutput(current_stdout_lines, failOn)
                if fail:
                    return False
                
                fail = failJobOnOutput(current_stderr_lines, failOn)
                if fail:
                    return False

                return True

            time.sleep(10)
            check_count += 1

    except Exception as e:
        print(f"Error monitoring job: {e}")
        exit()

def kill_job(job_id):
    """Kill a job using qdel"""
    try:
        cmd = f"qdel {job_id}"
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            print(f"Job {job_id} killed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error killing job: {e.output.decode().strip()}")
    except Exception as e:
        print(f"Error killing job: {e}")

if __name__ == "__main__":
    n = 0
    while True:
        jobId = launchJob()
        result = monitor_job(jobId, verbose=False, failOn=[("TypeError", 2), ("Blazegraph may already be running", 1)], nStdoutLines=90)

        if result == True:
            print(f"Job launched successfully! Sleeping for 2 hours...")
            time.sleep(60 * 60 * 2)  # Sleep for 2 hours
            kill_job(jobId)
            time.sleep(20)
            n += 1
            print(f"Rerunning job ({n})")
        else:
            n += 1
            print(f"Job failed! Rerunning job ({n})")
            kill_job(jobId)
            time.sleep(20)


