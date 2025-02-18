import time 

print('This script is running in Docker')

for i in range(5):
    print(f'Processing Step:{i + 1}') 
    time.sleep(1) 

print('Script Evaluation Completed...') 