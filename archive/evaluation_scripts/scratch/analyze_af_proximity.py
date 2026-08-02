
import re
from collections import defaultdict

def analyze_af_door_proximity(log_file):
    af_distances = defaultdict(list)
    
    # Pattern to match [NEW_LOG_V3] lines
    # Example: [NEW_LOG_V3] PASS-A(NF) | item= 165 (Synthetic red meat 90) NF wt=16.1 prio=1 af=10 zone=1 xyz=( 1.77, 3.24, 0.00) ml=( 1.77, 3.24) d_door= 2.22m ...
    pattern = re.compile(r'af=\s*(\d+).*d_door=\s*([\d\.]+)m')
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                af = int(match.group(1))
                dist = float(match.group(2))
                af_distances[af].append(dist)
                
    print(f"{'AF':<5} | {'Count':<6} | {'Avg Dist':<10}")
    print("-" * 30)
    for af in sorted(af_distances.keys(), reverse=True):
        dists = af_distances[af]
        avg_dist = sum(dists) / len(dists)
        print(f"{af:<5} | {len(dists):<6} | {avg_dist:<10.2f}m")

if __name__ == "__main__":
    analyze_af_door_proximity('placement_debug.log')
