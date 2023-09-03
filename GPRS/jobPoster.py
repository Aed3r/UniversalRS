from fnmatch import fnmatch
import sys
import paramiko
import time
import os
import pathlib
from stat import S_ISDIR
import math

# Workspace folder, don't change this
HERE = pathlib.Path(__file__).resolve().parent

# Auth
DEFAULT_HOSTNAME = 'gemini.science.uu.nl'
DEFAULT_USERNAME = '1060546'
DEFAULT_KEY_FILE = HERE.parent.joinpath('id_rsa')

# Default folders and files (local and remote)
DEFAULT_REQUIREMENTS_FILE = './requirements.txt'
DEFAULT_SYNC_FOLDER = './GPRS'
REMOTE_LOGS_LOC = './logs'
REMOTE_TMP_LOC = './tmp'

# Sync settings
DEFAULT_IGNORE_PATTERN = ['*__pycache__*']

# Script settings
DEFAULT_SCRIPT_NAME = 'job_script.sh'

# Job settings
DEFAULT_COMMAND = 'python3'

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

class SSHGridEngine:
    def __init__(self, username, hostname=DEFAULT_HOSTNAME, key_file=DEFAULT_KEY_FILE, port=22):
        self.hostname = hostname
        self.username = username
        self.key_file = key_file
        self.port = port
        self.client = None
        self.sftp = None

    def connect(self):
        """Connect to the SSH server."""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            private_key = paramiko.RSAKey(filename=self.key_file)
            self.client.connect(self.hostname, port=self.port, username=self.username, pkey=private_key)
            self.sftp = self.client.open_sftp()
        except Exception as e:
            print(f"Error connecting to SSH server: {e}")
        
        print("Connected to SSH server successfully!")

    def _should_ignore(self, path, ignore_patterns):
        """Check if a file or folder should be ignored."""
        for pattern in ignore_patterns:
            if fnmatch(path, pattern):
                return True
        return False
    
    def _is_remote_directory(self, path):
        """Check if a remote path is a directory."""
        try:
            return S_ISDIR(self.sftp.stat(path).st_mode)
        except Exception:
            return False

    def synchronize_files(self, local_folder, remote_folder, ignore_patterns=['*__pycache__*'], verbose=True, delete=True):
        """
        Synchronize a local folder with a remote folder on an SSH server.

        Parameters:
        - local_folder (str): The path to the local folder that needs to be synchronized.
        - remote_folder (str): The path on the remote SSH server where the contents should be synchronized to.
        - ignore_patterns (list of str, optional): A list of patterns that determine which files or folders should be ignored 
        during synchronization. The patterns support wildcards (`*` and `?`).
        - verbose (bool, optional): Whether or not to print out status messages during the synchronization process.
        - delete (bool, optional): Whether or not to delete files on the remote server that are not present on the local system.

        Behavior:
        - The function ensures that the structure and content of the `local_folder` are mirrored on the SSH server under `remote_folder`.
        - Any files or directories in `local_folder` that are not in `remote_folder` will be uploaded.
        - Any files in `remote_folder` that are not in `local_folder` will be removed.
        - If a file's modification time on the local system is more recent than that on the remote server, it will be re-uploaded.
        - Directories and subdirectories are handled recursively, ensuring the entire folder structure is mirrored.
        - Files or folders that match any pattern in `ignore_patterns` will not be synchronized.

        Returns:
        - None, but prints out status messages and errors during the synchronization process.

        Example:
        >>> sshge = SSHGridEngine('hostname', 'username', 'path/to/key')
        >>> sshge.connect()
        >>> sshge.synchronize_files('./local_path', './remote_path', ['*__pycache__*', '*.log'])
        Files synchronized successfully!
        >>> sshge.close()
        """

        try:
            # Replace Windows path separators with Unix path separators
            remote_folder = remote_folder.replace('\\', '/')
            
            # Check if remote folder exists
            try:
                self.sftp.stat(remote_folder)
            except FileNotFoundError:
                self.sftp.mkdir(remote_folder)

            # List local and remote files
            local_entries = set(os.listdir(local_folder))
            remote_entries = set(self.sftp.listdir(remote_folder))

            # Upload new or modified files
            for entry in local_entries:
                local_entry_path = os.path.join(local_folder, entry)
                remote_entry_path = os.path.join(remote_folder, entry).replace('\\', '/')

                # Check if entry should be ignored
                if self._should_ignore(local_entry_path, ignore_patterns):
                    if verbose:
                        print(f"Ignoring {local_entry_path}")
                    continue

                # If entry is a directory, handle recursively
                if os.path.isdir(local_entry_path):
                    self.synchronize_files(local_entry_path, remote_entry_path, ignore_patterns)
                else:
                    # Check modification times to decide if we should upload
                    should_upload = False
                    if entry not in remote_entries:
                        should_upload = True
                    else:
                        local_mtime = os.path.getmtime(local_entry_path)
                        remote_mtime = self.sftp.stat(remote_entry_path).st_mtime
                        if local_mtime > remote_mtime:
                            should_upload = True

                    if should_upload:
                        self.sftp.put(local_entry_path, remote_entry_path)
                        print(f"Uploaded {local_entry_path} to {remote_entry_path}")

            if delete:
                for entry in remote_entries - local_entries:
                    remote_entry_path = os.path.join(remote_folder, entry).replace('\\', '/')  # Adjust the separator here
                    if self._is_remote_directory(remote_entry_path):
                        self.sftp.rmdir(remote_entry_path)
                    else:
                        self.sftp.remove(remote_entry_path)
                    print(f"Removed {remote_entry_path}")
            
            if verbose:
                print("Files in '" + local_folder + "' synchronized successfully!")
        except Exception as e:
            print(f"Error synchronizing files: {e}")

    def create_job_script(self, command_file, command=DEFAULT_COMMAND, script_name=DEFAULT_SCRIPT_NAME, script_folder=DEFAULT_SYNC_FOLDER):
        """Create a .sh script with the provided command."""
        try:
            command_file_location = os.path.join(script_folder, command_file).replace('\\', '/')
            command = f"""{command} {command_file_location}"""

            script_content = f"""#!/bin/bash
echo "Executing command: {command}"
START_TIME=$(date +%s)
{command}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Command executed in $DURATION seconds"
"""
            script_location = os.path.join(script_folder, script_name).replace('\\', '/')

            # Write script content to a local file in binary mode with UTF-8 encoding
            with open(script_location, 'wb') as f:
                f.write(script_content.encode('utf-8'))

            print(f"Job script '{script_location}' created successfully!")
        except Exception as e:
            print(f"Error creating job script: {e}")

    def install_requirements(self, requirements_path="./requirements.txt"):
        """Install Python packages from requirements.txt on the remote server."""
        try:
            remote_requirements_path = os.path.join(REMOTE_TMP_LOC, 'requirements.txt').replace('\\', '/')

            # Check if remote folder exists
            try:
                self.sftp.stat(REMOTE_TMP_LOC)
            except FileNotFoundError:
                self.sftp.mkdir(REMOTE_TMP_LOC)
            
            # Check if the local requirements file has changed compared to the remote one
            local_mtime = os.path.getmtime(requirements_path)
            try:
                remote_mtime = self.sftp.stat(remote_requirements_path).st_mtime
            except FileNotFoundError:
                remote_mtime = 0
            
            # If the local file is newer, synchronize it and install the requirements
            if local_mtime > remote_mtime:
                self.sftp.put(requirements_path, remote_requirements_path)
                
                # Install packages
                cmd = f"pip install -r {remote_requirements_path}"
                _, stdout, stderr = self.client.exec_command(cmd)
                output = stdout.read().decode()
                error = stderr.read().decode()

                if error:
                    print(f"Error installing requirements: {error}")
                else:
                    print(f"Requirements installed successfully:\n{output}")
            else:
                print("Requirements have not changed. Skipping installation.")
                
        except Exception as e:
            print(f"Error during the requirements installation process: {e}")


    def submit_job(self, script_name=DEFAULT_SCRIPT_NAME, script_folder=DEFAULT_SYNC_FOLDER):
        """Submit the job using qsub or equivalent."""
        try:
            # Check if remote logs folder exists
            try:
                self.sftp.stat(REMOTE_LOGS_LOC)
            except FileNotFoundError:
                self.sftp.mkdir(REMOTE_LOGS_LOC)
            
            script_loc = os.path.join(script_folder, script_name).replace('\\', '/')
            sge_root = "/data/ge2011.11 "  # Replace with the value you got from `echo $SGE_ROOT`
            cmd = f"export SGE_ROOT={sge_root}; /data/ge2011.11/bin/linux-x64/qsub -o {REMOTE_LOGS_LOC} -e {REMOTE_LOGS_LOC} {script_loc}"
            _, stdout, stderr = self.client.exec_command(cmd)
            output = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                print(f"Error submitting job: {error}")
                return None
            else:
                job_id = output.split()[2]  # Extract job ID from qsub output
                print(f"Job {job_id} submitted successfully!")
                return job_id
        except Exception as e:
            print(f"Error submitting job: {e}")
            return None

    def monitor_job(self, job_id, job_name=DEFAULT_SCRIPT_NAME):
        """Periodically check job status using qstat, retrieve and print execution results."""
        check_count = 1
        last_stdout_lines = []  # A list to track the last printed lines from stdout
        last_stderr_lines = []  # A list to track the last printed lines from stderr
        add_newline = False  # To add a newline between the last output and the new one
        first_output_line_printed = False  # Flag to indicate when the first line of the output is printed
        stdout_location = os.path.join(REMOTE_LOGS_LOC, f"{job_name}.o{job_id}").replace("\\", "/")
        stderr_location = os.path.join(REMOTE_LOGS_LOC, f"{job_name}.e{job_id}").replace("\\", "/")

        def print_updated_lines(current_output_lines, last_lines):
            """Helper function to print the updated lines."""
            # If the number of lines in the current output is the same as the last output
            if len(current_output_lines) == len(last_lines) and current_output_lines:
                # If the last line has changed, update it
                if current_output_lines[-1] != last_lines[-1]:
                    print("\033[F", end='')  # Move the cursor up one line
                    print("\r" + current_output_lines[-1].ljust(len(last_lines[-1])), end='', flush=True)
                    last_lines[-1] = current_output_lines[-1]

            # If there's a new line added, print it and update the last line
            elif len(current_output_lines) > len(last_lines):
                # Calculate how many lines to move the cursor up
                move_up = len(last_lines) - len(current_output_lines) + 1

                # Move the cursor up
                print(f"\033[{move_up}F", end='')

                # Print all the new lines
                for line in current_output_lines[len(last_lines):]:
                    print("\r" + line, end='', flush=True)
                    print("\033[E", end='')  # Move the cursor to the next line

                last_lines = current_output_lines

            return last_lines
        
        try:
            
            while True:
                sge_root = "/data/ge2011.11 "  # Replace with the value you got from `echo $SGE_ROOT`
                cmd = f"export SGE_ROOT={sge_root}; /data/ge2011.11/bin/linux-x64/qstat"
                _, stdout, stderr = self.client.exec_command(cmd)
                output_lines = stdout.read().decode().splitlines()

                # Extract line corresponding to the job of interest
                job_line = next((line for line in output_lines if line.strip().startswith(str(job_id))), None)

                if job_line:
                    state = job_line.split()[4]
                    
                    if state in ['r', 'Rr']:  # If the job is in a running state
                        # Retrieve the current output from the job_script.sh.o<id> file
                        _, stdout, _ = self.client.exec_command(f"cat {stdout_location}")
                        current_stdout_lines = stdout.read().decode().splitlines()

                        # Retrieve the current output from the job_script.sh.e<id> file
                        _, stderr, _ = self.client.exec_command(f"cat {stderr_location}")
                        current_stderr_lines = stderr.read().decode().splitlines()

                        # If it's the first output line and it hasn't been printed yet
                        if not first_output_line_printed and current_stdout_lines:
                            print(current_stdout_lines[0])
                            first_output_line_printed = True
                            last_stdout_lines.append(current_stdout_lines[0])
                            add_newline = True
                            continue

                        # Print updated lines for stdout
                        last_stdout_lines = print_updated_lines(current_stdout_lines, last_stdout_lines)

                        # Print updated lines for stderr
                        last_stderr_lines = print_updated_lines(current_stderr_lines, last_stderr_lines)
                    else:
                        if add_newline:
                            print()
                            add_newline = False
                        
                        print(f"Job {job_id} is in state: {JOB_STATES.get(state, str(state))}")

                else:
                    if add_newline:
                            print()
                            add_newline = False
                    if first_output_line_printed:
                        print(f"Job {job_id} has completed!")
                    else:
                        print(f"Job {job_id} has completed or does not exist!")

                    # Retrieve the results from output and error files
                    for ext in ["o", "e"]:
                        file_name = os.path.join(REMOTE_LOGS_LOC, f"job_script.sh.{ext}{job_id}").replace("\\", "/")
                        _, stdout, _ = self.client.exec_command(f"cat {file_name}")
                        file_content = stdout.read().decode()
                        if file_content:
                            print(f"Contents of {file_name}:")
                            print(file_content)
                    break

                time.sleep(5)
                check_count += 1

        except Exception as e:
            print(f"Error monitoring job: {e}")

    def run_python_job(self, command_file, command=DEFAULT_COMMAND, folder_path=DEFAULT_SYNC_FOLDER, script_name=DEFAULT_SCRIPT_NAME, requirements_path=DEFAULT_REQUIREMENTS_FILE, ignore_pattern=DEFAULT_IGNORE_PATTERN):
        """Execute the entire workflow of synchronizing, creating a job, and monitoring."""
        try:
            # Synchronize files
            self.synchronize_files(folder_path, folder_path, ignore_pattern)
            print(f"Sucessfully synchronized '{folder_path}'.")

            # Create job script
            self.create_job_script(command_file, command, script_name, folder_path)

            # Install requirements
            self.install_requirements(requirements_path)

            # Submit job
            job_id = self.submit_job(script_name, folder_path)
            if not job_id:
                return

            # Monitor job
            self.monitor_job(job_id, script_name)
        except Exception as e:
            print(f"Error running python job: {e}")

    def close(self):
        """Close the SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()

# Example usage
# sshge = SSHGridEngine('1060546')
# sshge.connect()
# sshge.run_python_job('/local/folder', '/local/path/requirements.txt', 'python my_script.py')
# sshge.close()

if __name__ == "__main__":
    sshge = SSHGridEngine(DEFAULT_USERNAME)

    # Check if arguments are specified
    if len(sys.argv) > 1:
        option = sys.argv[1].strip()

        if option.strip() == "-s" or option.strip() == "--sync":
            sshge.connect()
            if len(sys.argv) > 2:
                folder_path = sys.argv[2].strip()

                if len(sys.argv) > 3:
                    ignore_pattern = sys.argv[3].strip()
                    sshge.synchronize_files(folder_path, folder_path, ignore_pattern)
                else:
                    sshge.synchronize_files(folder_path, folder_path)
            else:
                sshge.synchronize_files(DEFAULT_SYNC_FOLDER, DEFAULT_SYNC_FOLDER)
            print(f"Synchronized '{folder_path}'.")
        elif option.strip() == "-i" or option.strip() == "--install":
            sshge.connect()
            if len(sys.argv) > 2:
                requirements_path = sys.argv[2].strip()
                sshge.install_requirements(requirements_path)
            else:
                sshge.install_requirements(DEFAULT_REQUIREMENTS_FILE)
        elif option.strip() == "-c" or option.strip() == "--create":
            if len(sys.argv) > 2:
                command_file = sys.argv[2].strip()

                if len(sys.argv) > 3:
                    command = sys.argv[3].strip()
                    
                    if len(sys.argv) > 4:
                        script_folder = sys.argv[4].strip()

                        if len(sys.argv) > 5:
                            script_name = sys.argv[5].strip()

                            sshge.create_job_script(command_file, command, script_name, script_folder)
                        else:
                            sshge.create_job_script(command_file, command, DEFAULT_SCRIPT_NAME, script_folder)
                    else:
                        sshge.create_job_script(command_file, command, DEFAULT_SCRIPT_NAME, DEFAULT_SYNC_FOLDER)
                else:
                    sshge.create_job_script(command_file, DEFAULT_COMMAND, DEFAULT_SCRIPT_NAME, DEFAULT_SYNC_FOLDER)
            else:
                print(f"Please specify a python file located in '{DEFAULT_SYNC_FOLDER}' to run (eg 'test.py').")
        elif option.strip() == "-q" or option.strip() == "--queue":
            sshge.connect()
            if len(sys.argv) > 2:
                script_name = sys.argv[2].strip()

                if len(sys.argv) > 3:
                    script_folder = sys.argv[3].strip()

                    sshge.submit_job(script_name, script_folder)
                else:
                    sshge.submit_job(script_name, DEFAULT_SYNC_FOLDER)
            else:
                sshge.submit_job(DEFAULT_SCRIPT_NAME, DEFAULT_SYNC_FOLDER)
        elif option.strip() == "-m" or option.strip() == "--monitor":
            if len(sys.argv) > 2:
                sshge.connect()
                job_id = sys.argv[2].strip()

                if len(sys.argv) > 3:
                    script_name = sys.argv[3].strip()

                    sshge.monitor_job(job_id, script_name)
                else:
                    sshge.monitor_job(job_id, DEFAULT_SCRIPT_NAME)
            else:
                print("Please specify a job ID to monitor.")
        elif option.strip() == "-r" or option.strip() == "--run":
            if len(sys.argv) > 2:
                sshge.connect()
                command_file = sys.argv[2].strip()

                if len(sys.argv) > 3:
                    command = sys.argv[3].strip()

                    if len(sys.argv) > 4:
                        folder_path = sys.argv[4].strip()

                        if len(sys.argv) > 5:
                            script_name = sys.argv[5].strip()

                            if len(sys.argv) > 6:
                                requirements_path = sys.argv[6].strip()

                                if len(sys.argv) > 7:
                                    ignore_pattern = sys.argv[7].strip()

                                    sshge.run_python_job(command_file, command, folder_path, script_name, requirements_path, ignore_pattern)
                                else:
                                    sshge.run_python_job(command_file, command, folder_path, script_name, requirements_path, DEFAULT_IGNORE_PATTERN)
                            else:
                                sshge.run_python_job(command_file, command, folder_path, script_name, DEFAULT_REQUIREMENTS_FILE, DEFAULT_IGNORE_PATTERN)
                        else:
                            sshge.run_python_job(command_file, command, folder_path, DEFAULT_SCRIPT_NAME, DEFAULT_REQUIREMENTS_FILE, DEFAULT_IGNORE_PATTERN)
                    else:
                        sshge.run_python_job(command_file, command, DEFAULT_SYNC_FOLDER, DEFAULT_SCRIPT_NAME, DEFAULT_REQUIREMENTS_FILE, DEFAULT_IGNORE_PATTERN)
                else:
                    sshge.run_python_job(command_file, DEFAULT_COMMAND, DEFAULT_SYNC_FOLDER, DEFAULT_SCRIPT_NAME, DEFAULT_REQUIREMENTS_FILE, DEFAULT_IGNORE_PATTERN)
            else:
                print(f"Please specify a python file located in '{DEFAULT_SYNC_FOLDER}' to run (eg 'test.py').")
        elif option.strip() == "-h" or option.strip() == "--help" or option.strip() == "":
            print("Usage: python jobPoster.py [option] [arguments]")
            print("Options:")
            print("  -s, --sync [folder] [ignore_pattern]                                                             Synchronize files in the specified folder. If no folder is specified, the default folder is used.")
            print("  -i, --install [requirements_file]                                                                Install the packages specified in the requirements file. If no file is specified, the default file is used.")
            print("  -c, --create [command_file] [command] [script_folder] [script_name]                              Create a job script with the specified command. If no command is specified, the default command is used. If no script folder is specified, the default script folder is used. If no script name is specified, the default script name is used.")
            print("  -q, --queue [script_name] [script_folder]                                                        Submit a job to the queue with the specified script name. If no script name is specified, the default script name is used. If no script folder is specified, the default script folder is used.")
            print("  -m, --monitor [job_id] [script_name]                                                             Monitor a job with the specified job ID and script name. If no script name is specified, the default script name is used.")
            print("  -r, --run <command_file> [command] [folder_path] [script_name] [requirements_path] [ignore_path] Synchronizes files and requirements, then creates, submits and monitors a job for the specified python file. If any arguments are not specified other than the command_file, the default values are used.")
            print("  -h, --help                                                                                       Display this help message.")
            print("  [python file | job script | folder]                                                              Runs the specified python file or job script. This will synchronize the local folder they are in. If a folder is specified it will only be synchronized.")
        else:
            arg = sys.argv[1].strip()

            # Check if the file is a python script
            if arg.find(".py") != -1:
                # Check if the file is in a folder
                if arg.find("/") != -1:
                    command_file = arg[arg.rfind("/") + 1:]
                    sync_folder = arg[:arg.rfind("/")]
                elif arg.find("\\") != -1:
                    command_file = arg[arg.rfind("\\") + 1:]
                    sync_folder = arg[:arg.rfind("\\")]
                else:
                    command_file = arg
                    sync_folder = DEFAULT_SYNC_FOLDER
                
                # Check that the sync folder exists in the current directory
                if not os.path.exists(os.path.join(HERE.parent, sync_folder)):
                    print(f"Error: cannot find '{sync_folder}' in workspace '{HERE.parent}'. Please specify a valid folder located inside the current directory.")
                    exit()
                
                # Check that the file exists
                if not os.path.exists(os.path.join(sync_folder, command_file)):
                    print(f"Error: cannot find '{arg}' in '{sync_folder}'. Please specify a valid file or specify the sync folder containing that file.")
                    exit()

                sshge.connect()
                sshge.run_python_job(
                    command_file=command_file, 
                    command=DEFAULT_COMMAND, 
                    folder_path=sync_folder, 
                    script_name=DEFAULT_SCRIPT_NAME, 
                    requirements_path=DEFAULT_REQUIREMENTS_FILE,
                    ignore_pattern=DEFAULT_IGNORE_PATTERN)
            # Check if the file is a job
            elif arg.find(".sh") != -1:
                # Check if the file is in a folder
                if arg.find("/") != -1:
                    script_name = arg[arg.rfind("/") + 1:]
                    sync_folder = arg[:arg.rfind("/")]
                elif arg.find("\\") != -1:
                    script_name = arg[arg.rfind("\\") + 1:]
                    sync_folder = arg[:arg.rfind("\\")]
                else:
                    script_name = arg
                    sync_folder = DEFAULT_SYNC_FOLDER
                
                # Check that the sync folder exists in the current directory
                if not os.path.exists(os.path.join(HERE.parent, sync_folder)):
                    print(f"Error: cannot find '{sync_folder}' in workspace '{HERE.parent}'. Please specify a valid folder located inside the current directory.")
                    exit()
                
                # Check that the file exists
                if not os.path.exists(os.path.join(sync_folder, script_name)):
                    print(f"Error: cannot find '{arg}' in '{sync_folder}'. Please specify a valid file or specify the sync folder containing that file.")
                    exit()

                sshge.connect()
                sshge.synchronize_files(sync_folder, sync_folder, DEFAULT_IGNORE_PATTERN)
                print(f"Synchronized '{sync_folder}'.")
                sshge.install_requirements(DEFAULT_REQUIREMENTS_FILE)
                job = sshge.submit_job(script_name, sync_folder)
                if job is not None:
                    sshge.monitor_job(job, script_name)
            else:
                # Check that the sync folder exists in the current directory
                if not os.path.exists(os.path.join(HERE.parent, arg)):
                    print(f"Error: cannot find '{arg}' in workspace '{HERE.parent}'. Please specify a valid folder located inside the current directory.")
                    exit()
                
                sshge.connect()
                sshge.synchronize_files(arg, arg, DEFAULT_IGNORE_PATTERN)
                print(f"Synchronized '{arg}'.")
    else:
        print("Usage: python jobPoster.py -h")


    # sshge.create_job_script('python3 GPRS/test.py', 'GPRS/job_script.sh')
    # sshge.synchronize_files('GPRS', 'GPRS')
    # sshge.install_requirements('./requirements.txt')
    # job = sshge.submit_job("GPRS/job_script.sh")
    # sshge.monitor_job(job)
    #sshge.run_python_job(folder_path='GPRS', requirements_path='./requirements.txt', command='python3 GPRS/test.py')
    sshge.close()