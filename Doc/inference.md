# Here are some basic Documenation of Airvat

## 1. check the `PARTITION AVAIL`  `TIMELIMIT`  `NODES`  `STATE NODELIST`
```
$sinfo
```


### This is the output of our inference.
```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
dgxnp*       up 7-00:00:00     19    mix scn2-10g,scn5-10g,scn6-10g,scn8-10g,scn9-10g,scn14-10g,scn17-10g,scn20-10g,scn21-10g,scn23-10g,scn24-10g,scn25-10g,scn29-10g,scn31-10g,scn32-10g,scn33-10g,scn36-10g,scn37-10g,scn41-10g
dgxnp*       up 7-00:00:00      6  alloc scn1-10g,scn3-10g,scn11-10g,scn18-10g,scn28-10g,scn30-10g
cpup         up 7-00:00:00     20    mix scn2-10g,scn5-10g,scn6-10g,scn8-10g,scn9-10g,scn14-10g,scn17-10g,scn20-10g,scn21-10g,scn23-10g,scn24-10g,scn25-10g,scn29-10g,scn31-10g,scn32-10g,scn33-10g,scn34-10g,scn36-10g,scn37-10g,scn41-10g
cpup         up 7-00:00:00      6  alloc scn1-10g,scn3-10g,scn11-10g,scn18-10g,scn28-10g,scn30-10g
```
This output gives us the exact missing puzzle piece for your Slurm script. It shows all the available "queues" (partitions) on your cluster and what the computers (nodes) inside them are currently doing.

Here is the breakdown of what you have:

### **0.1. The Partitions (Your Queues)**

* **`dgxnp*`**: This is your GPU partition. "DGX" is NVIDIA’s enterprise machine that contains exactly 8 GPUs (which perfectly matches your A100 `nvidia-smi` output). The asterisk (`*`) means this is the **default** partition. If you don't specify a partition in your script, Slurm will automatically send it here.
* **`cpup`**: This is likely a partition dedicated to CPU-only tasks. You will ignore this one for your multi-GPU inference.

### **0.2. Node States (`mix` vs `alloc`)**

* **`mix`**: The node is partially in use, but still has some resources (CPUs, GPUs, or memory) available. You have 19 nodes in this state.
* **`alloc`**: The node is fully allocated (100% busy) by other users. You have 6 nodes completely tied up.

### **0.3. Time Limit**

* **`7-00:00:00`**: This means the absolute maximum time you can request for a single job is **7 days**.

---

## 2. check the queue to verify your job is queued or running:

```
squeue -u $USER
```

## 3. Look under the `JOBID` column for the number assigned to your job.
```
scancel 12345
```

- Cancel it by typing `scancel` followed by that number. For example, if your Job ID is `12345`:

### or 
```
scancel -u $USER
```
