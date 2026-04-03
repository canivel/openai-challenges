# Kaos Restart Procedure

## Why Restart is Required
The Kaos MCP server reads `kaos.yaml` once at startup and caches the config.
After updating kaos.yaml to use OpenAI provider (gpt-4o), the MCP server must
be restarted to pick up the change. Until then, all `agent_spawn` calls fail
with "404 Not Found for localhost:8000".

## Step 1: Restart Claude Code
Close and reopen Claude Code (or reload the window in VS Code).
This restarts all MCP servers, which will re-read kaos.yaml.

## Step 2: Verify OPENAI_API_KEY is set
In terminal:
```bash
echo $OPENAI_API_KEY   # Linux/Mac
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY    # Windows PowerShell
```

## Step 3: Test Kaos backend
The first thing to do after restart is test the spawn:
```
Use mcp__kaos__agent_spawn with:
  name: "connectivity-test"
  task: 'Respond with exactly: "KAOS_OPENAI_OK: backend confirmed working". Nothing else.'
```
Expected result: agent completes with that exact string in its output.

## Step 4: Kill old initialized agents (optional, cleanup)
The following agents are in "initialized" state (never ran, just have VFS notes):
- 01KN86XZK4AHJ4VBGRW3TMMKHR (base-architect)
- 01KN86Y0RHS5PPK1J3J28ZTMJQ (slot-optimizer)
- 01KN86Y0SK45C65JEHRS9Q9FCA (quant-master)
- 01KN86Y0TMPSP5047VEK58JJNV (training-optimizer)
- 01KN86Y0VPWDCDMDXQZ1Q3BTVD (innovation-scout)

These have useful VFS content (autoresearch_task.md, research notes).
Do NOT kill them — read the task from their VFS when spawning new agents.

## Step 5: Spawn 5 new autoresearch agents in parallel
Use mcp__kaos__agent_parallel with these 5 agents simultaneously:

### base-architect
Task: "You are an architecture expert for OpenAI Parameter Golf.
Read f:\\Projects\\openai-challenges\\parameter-golf\\records\\our_submission\\train_gpt.py
and your task spec at VFS agent 01KN86XZK4AHJ4VBGRW3TMMKHR:/autoresearch_task.md (read via agent_read).
Propose ONE architecture change, implement it, write modified train_gpt.py to your /train_gpt_v2.py, document in /arch_results.md"

### slot-optimizer  
Task: "You are a SLOT eval expert for OpenAI Parameter Golf.
Read f:\\Projects\\openai-challenges\\parameter-golf\\records\\our_submission\\train_gpt.py
First determine if our SLOT implementation has causal violations (analyze eval_val_sliding_slot).
Then propose ONE improvement. Write variant to /slot_variant.py, document in /slot_results.md"

### quant-master
Task: "You are a quantization expert for OpenAI Parameter Golf.
Read f:\\Projects\\openai-challenges\\parameter-golf\\records\\our_submission\\train_gpt.py
Focus on quantize_int6_gptq() and the compression section.
Propose ONE quantization improvement. Write variant to /quant_variant.py, document in /quant_results.md"

### training-optimizer
Task: "You are a training hyperparameter expert for OpenAI Parameter Golf.
Read f:\\Projects\\openai-challenges\\parameter-golf\\records\\our_submission\\train_gpt.py
Focus on Hyperparameters class and main() training loop.
Propose ONE training improvement. Write variant to /training_variant.py, document in /training_results.md"

### innovation-scout
Task: "You are an innovation researcher for OpenAI Parameter Golf.
Read f:\\Projects\\openai-challenges\\parameter-golf\\records\\reference_prs\\pr1260_dexhunter_DECOMPRESSED.py
Find ONE technique not in our baseline at f:\\Projects\\openai-challenges\\parameter-golf\\records\\our_submission\\train_gpt.py
Write implementation to /innovation_variant.py, document in /innovation_results.md"

## Step 6: Query results when agents complete
```sql
-- Check what each agent produced
SELECT a.name, s.key, LENGTH(s.value) as content_size
FROM state s JOIN agents a ON s.agent_id = a.agent_id
WHERE s.key IN ('arch_results.md', 'slot_results.md', 'quant_results.md', 'training_results.md', 'innovation_results.md')
ORDER BY a.name;
```

## Step 7: Read each agent's results
Use agent_read for each agent to get the results.md and variant.py files.

## Step 8: Pick best variant and run on RunPod
Based on agent analysis:
1. Apply best changes to train_gpt.py
2. Do ONE CPU smoke test locally
3. Deploy to RunPod 8xH100 (~$8/run, budget allows 5 runs)

## What mh_search CANNOT Do For Us
mh_search requires a pre-registered benchmark (text_classify, math_rag, agentic_coding).
"parameter_golf" is NOT registered. To use it, we'd need to:
1. Implement a ParameterGolfBenchmark class in kaos/benchmarks/
2. Register it in kaos/benchmarks/__init__.py
3. The benchmark would need to run training and return BPB

This is NOT worth the effort for our use case. Use direct agent_spawn instead.
