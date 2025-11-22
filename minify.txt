# Python Minification Guide

## Mission
Reduce Python code by 30-60% tokens while maintaining identical functionality, valid syntax, and safe resource management.

## Priority Order (NEVER VIOLATE)
1. SAFETY -> Correctness, resource cleanup, error handling
2. SYNTAX -> Valid executable Python
3. COMPRESS -> Token reduction (only after 1 & 2 pass)

Golden Rule: When uncertain, preserve original. Extra tokens > broken code.

## Critical Safety Rules

### Rule 1: Indentation = Exactly 4 Spaces Per Level
```python
# Correct (0, 4, 8 spaces)
def foo():
    if x:
        process()

# Wrong - IndentationError
def foo():
  if x:      # 2 spaces
```

### Rule 2: Always Use Context Managers for File/Network/Database
```python
# WRONG - resource leak
data=json.load(open("f.json"))

# RIGHT - guaranteed cleanup
with open("f.json")as f:data=json.load(f)
```
Apply to: open(), sockets, databases, gzip.open(), zipfile.ZipFile(), any __enter__/__exit__ object

### Rule 3: Mandatory Spaces After Control Keywords
```python
# Must keep space after these:
if x    for i    while c    def f    class C    return x
import m    from m    as a    lambda x    in lst    is None
not x    and y    or z    elif x    else:    yield x    with f

# Can remove spaces around these:
=  ==  !=  +  -  *  /  [  ]  {  }  (  )  ,
```

### Rule 4: Remove All Type Annotations
```python
# Before
def process(data:list[int])->dict:
    result:dict={}

# After
def process(data):
    result={}
```

## Dangerous Patterns That Silently Fail

### CRITICAL: List Comprehensions Are For Building Data Only
Problem: Comprehensions execute but may fail silently with side effects.

```python
# WRONG - side effect in comprehension
[df.update({col:data[col]})for col in cols]  # Returns [None,None,...], update() may fail
[print(x)for x in items]  # Builds useless [None,None,...]
[db.insert(row)for row in data]  # No error handling

# RIGHT - explicit loop for side effects
for col in cols:df[col]=data[col]
for x in items:print(x)
for row in data:db.insert(row)
```

Safe comprehension usage:
```python
# Building data structures - OK
results=[transform(x)for x in items]
lookup={k:v*2 for k,v in data.items()}
unique={x.lower()for x in words}
```

Never use comprehensions for:
- Database operations (insert, update, delete)
- File I/O (write, read)
- Modifying external state (global vars, class attributes)
- Functions returning None
- Anything needing try/except

### CRITICAL: DataFrame.update() Only Modifies Existing Columns
Problem: Silently does nothing when columns don't exist yet.

```python
# WRONG - silently drops ALL data
df=pd.DataFrame(index=dates)
for name,series in data.items():
    df.update({name:series})  # Column doesn't exist - FAILS SILENTLY
# Result: Empty DataFrame, all data lost

# RIGHT - properly adds columns
df=pd.DataFrame(index=dates)
for name,series in data.items():
    df[name]=series

# ALSO RIGHT - build from dict
df=pd.DataFrame(data,index=dates)
```

When to use update(): Only when modifying values in columns that already exist. Never for adding new columns.

### CRITICAL: No break/continue in Comprehensions
```python
# WRONG - SyntaxError
[x for x in items if check(x,break)]

# RIGHT - use loop
for x in items:
    if condition:result=x;break
```

## Minification Process (Apply In Order)

### 1. Strip Non-Executing Content
Remove: comments, docstrings, type annotations, blank lines

### 2. Optimize Imports
```python
from module import func1,func2,Class1
import numpy as np,pandas as pd
```

### 3. Minimize Whitespace (Keep Mandatory Keyword Spaces)
```python
x=1+2
func(a,b,c)
data={"k":"v"}
lst=[1,2,3]
```

### 4. Single-Line Simple Statements
```python
if x>0:return x
for item in items:process(item)
```

### 5. Semicolons (Same Indentation Only)
```python
a=1;b=2;c=3  # Same level - OK

# Different levels - NO
if cond:
    x=1;y=2  # These can be combined
```

### 6. Apply Safe Transformations

Ternary:
```python
result=val1 if condition else val2
```

Default with or:
```python
value=arg or default
```

Walrus (3.8+):
```python
if(m:=pattern.search(text)):use(m)
```

Safe comprehensions (pure data building only):
```python
squares=[x*2 for x in nums]
lookup={k:v.upper()for k,v in items.items()}
```

Chain operations:
```python
return transform(get_data())
```

## Decision Rules (Apply These Tests)

Should I use a comprehension?
- Building list/dict/set? YES -> Consider comprehension
- Has side effects? YES -> NO comprehension, use loop
- Returns None? YES -> NO comprehension, use loop
- Needs try/except? YES -> NO comprehension, use loop
- Uses break/continue? YES -> NO comprehension, use loop

Should I use context manager?
- Uses open()? YES -> MUST use with statement
- Uses network/database? YES -> MUST use with statement
- Has __enter__/__exit__? YES -> MUST use with statement

Should I combine with semicolons?
- Same indentation? YES -> Can combine
- Simple statements? YES -> Can combine
- More than 3 statements? MAYBE -> Consider readability

Should I inline if/else?
- Simple assignment? YES -> Use ternary
- Simple return? YES -> Use ternary
- Multiple statements per branch? NO -> Keep multi-line

## Pre-Output Checklist (Must Pass All)

Check before outputting:
- Indentation is 4-space multiples
- All file operations use with statements
- Spaces after keywords (if, for, def, etc.) present
- No side effects in comprehensions
- Try/except blocks preserved
- Would execute without errors: python -m py_compile
- Identical functionality to original

If any check fails: Stop and revise. Do not output broken code.

## Complete Example

Before (23 lines):
```python
import json
from pathlib import Path

def load_and_process(filepath, items):
    """Load config and process items."""
    # Load configuration
    with open(filepath, 'r') as f:
        config = json.load(f)

    # Set default timeout
    if 'timeout' not in config:
        config['timeout'] = 30

    # Process items
    results = []
    for item in items:
        if item is not None:
            transformed = item * 2 if isinstance(item, int) else str(item)
            results.append(transformed)

    return config, results
```

After (7 lines, 70% reduction):
```python
import json
from pathlib import Path
def load_and_process(filepath,items):
    with open(filepath,'r')as f:config=json.load(f)
    if 'timeout'not in config:config['timeout']=30
    results=[item*2 if isinstance(item,int)else str(item)for item in items if item is not None]
    return config,results
```

## Common Mistakes (Never Do These)

```python
# WRONG: Side effect in comprehension
[db.insert(row)for row in data]
[file.write(line)for line in lines]
[df.update({k:v})for k,v in cols.items()]

# WRONG: DataFrame.update() for new columns
df=pd.DataFrame()
df.update({"new_col":series})  # Does nothing

# WRONG: No context manager
config=json.load(open("f.json"))

# WRONG: Wrong indentation
def f():
  return 1  # 2 spaces

# WRONG: break in comprehension
[x for x in items if(check(x),break)[0]]

# RIGHT: Explicit loops for side effects
for row in data:db.insert(row)
for line in lines:file.write(line)
for k,v in cols.items():df[k]=v

# RIGHT: Direct column assignment
df=pd.DataFrame()
df["new_col"]=series

# RIGHT: Context manager
with open("f.json")as f:config=json.load(f)

# RIGHT: 4-space indentation
def f():
    return 1

# RIGHT: Loop with break
for x in items:
    if condition:result=x;break
```

## Output Format

1. Minified code in code block
2. "Reduced from X to Y lines (Z% reduction)"
3. Note critical preservations if relevant

## Final Reminder

Safety > Syntax > Compression. Always.
Better to preserve 10 extra tokens than break 1 line of code.
