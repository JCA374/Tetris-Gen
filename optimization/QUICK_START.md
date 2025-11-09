# Quick Start Guide - Optimal Training Configuration

## Current System: DESKTOP-APPKIAQ (8 cores)

### Recommended Command (START HERE)

```bash
python src/train.py --population 28 --generations 100 --games 3 --workers 4
```

**Estimated time**: ~7.3 hours
**Why this config**: Perfect worker alignment (28÷4=7), excellent genetic diversity, deep optimization

---

## Alternative Configurations

### Quick Test (~2.5h)
```bash
python src/train.py --population 20 --generations 50 --games 3 --workers 4
```

### Extended Quality (~7.8h)
```bash
python src/train.py --population 32 --generations 100 --games 3 --workers 4
```

### Maximum Quality Overnight (~16h)
```bash
python src/train.py --population 40 --generations 100 --games 5 --workers 4
```

---

## Key Rules for This System

1. **Always use 4 workers** (optimal from benchmark)
2. **Population must be divisible by 4** for 100% efficiency
3. **Good populations**: 20, 24, 28, 32, 36, 40
4. **Bad populations**: 5, 10, 15, 25, 30, 35, 50

---

## Running on Different Computer (Same CPU, Less RAM)

If the other computer has <4GB available RAM, reduce workers:

### With 2 workers:
```bash
python src/train.py --population 20 --generations 100 --games 3 --workers 2
```
Use populations divisible by 2: 10, 12, 14, 16, 18, 20, 22, 24...

### With 1 worker (safest for low RAM):
```bash
python src/train.py --population 28 --generations 100 --games 3 --workers 1
```
Any population size works, but slower execution.

---

## To Optimize for a NEW System

1. Run benchmark:
   ```bash
   python src/benchmark.py
   ```

2. Note optimal worker count from output

3. Analyze worker efficiency:
   ```bash
   python optimization/analyze_worker_efficiency.py --workers <OPTIMAL_WORKERS>
   ```

4. Choose population that's divisible by worker count

5. Create new config file:
   - Copy `system_config_DESKTOP-APPKIAQ_8CORE.json`
   - Update with new system details
   - Name it: `system_config_<HOSTNAME>_<CORES>CORE.json`

---

## Summary Cheat Sheet

| What | Value | Why |
|------|-------|-----|
| Workers | 4 | From benchmark (6.96× speedup) |
| Population | 28 | Divisible by 4, good diversity |
| Generations | 100 | Deep optimization |
| Games | 3 | Balance accuracy/speed |
| Expected Time | ~7-8h | Based on 0.34 games/sec |

**Current broken config (population=5)**: Only 62.5% efficient, insufficient diversity ❌
**Recommended config (population=28)**: 100% efficient, excellent diversity ✓
