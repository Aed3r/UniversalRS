# Program that sleeps for 10 seconds and then prints "Hello World"

import sys
from tqdm import tqdm
import time
 
for i in tqdm(range(int(4))):
    time.sleep(1)

print("Hello World")