**`declarative-py` - Business Logic as YAML**

> **Replace Python boilerplate with declarative rules. Get AI fallback for free.**

The zero-dependency Python library that turns complex frameworks into simple YAML.

---

## 🚀 **What It Does**

**Instead of this:**
```python
from pydantic import BaseModel
from fastapi import FastAPI
from langchain.llms import OpenAI

class User(BaseModel):
    name: str
    age: int

@app.post("/validate")
def validate_user(data: dict):
    try:
        user = User(**data)
        return {"valid": True}
    except ValidationError:
        # Call LLM to fix data
        llm = OpenAI()
        fixed = llm(f"Fix this user data: {data}")
        return {"valid": False, "fixed": fixed}
```

**Write this:**
```python
from declarative_py import validate, fallback

@fallback(llm="gpt-4", prompt="Fix this user data: {input}")
def validate_user(data):
    is_valid, _ = validate(data, {"name": "str", "age": "int"})
    return {"valid": is_valid}
```

---

## 🎯 **Who It's For**

| Persona | Their Pain | declarative-py Solution |
|---------|------------|------------------------|
| **Data Scientists** | Pydantic boilerplate for every schema | `validate(data, {"column": "type"})` |
| **API Developers** | FastAPI decorators, dependency injection | `RuleEngine("rules.yaml").serve(8000)` |
| **AI Engineers** | LangChain complexity for simple LLM calls | `@fallback` decorator |
| **Business Analysts** | Can't deploy Python logic | YAML rules that run anywhere |

---

## 💡 **Core Features**

### **1. Schema Validation Without Classes**
```python
# No more Pydantic models
schema = {"name": "str", "age": "int?", "email": "str"}
is_valid, errors = validate(data, schema)
```

### **2. YAML-Powered Business Rules**
```yaml
# rules/pricing.yaml
rules:
  - if: "customer_type == 'vip' and order_amount > 1000"
    then:
      discount: 0.15
      shipping: "free"
  - else:
    then:
      discount: 0.05
```

```python
engine = RuleEngine("rules/pricing.yaml")
result = engine.run(order_data)  # {"discount": 0.15, "shipping": "free"}
```

### **3. AI Fallback Built-In**
```python
@fallback(llm="gpt-4", prompt="Extract entities from: {input}")
def extract_entities(text):
    # If your function fails, AI takes over
    return my_parser(text)
```

### **4. Instant APIs**
```python
# Zero to REST API in one line
RuleEngine("business_rules.yaml").serve(port=8000)
```

---

## 🛠 **Zero Dependencies, Maximum Power**

**What you get:**
- ✅ **Validation** (replaces Pydantic)
- ✅ **Rule Engine** (replaces complex conditionals)  
- ✅ **HTTP Server** (replaces FastAPI for simple APIs)
- ✅ **LLM Integration** (replaces LangChain for basic use cases)
- ✅ **Type Checking** (runtime validation with coercion)

**What you don't get:**
- ❌ **No dependencies** (stdlib only, optional requests/yaml)
- ❌ **No boilerplate** (write logic, not configuration)
- ❌ **No framework lock-in** (just functions)

---

## 🎪 **Real-World Usage**

### **Data Validation Pipeline**
```python
from declarative_py import validate, RuleEngine, fallback

@fallback(prompt="Clean and validate this customer data: {input}")
def process_customer(raw_data):
    # Validate structure
    is_valid, _ = validate(raw_data, {
        "name": "str", 
        "email": "str", 
        "age": "int?"
    })
    
    # Apply business rules
    engine = RuleEngine("rules/customer_tier.yaml")
    return engine.run(raw_data)
```

### **Instant Business API**
```python
# Deploy rules as REST API in 30 seconds
RuleEngine("rules/loan_approval.yaml").serve(port=8000)

# Now POST JSON to http://localhost:8000
# {"income": 75000, "credit_score": 680} 
# → {"approved": true, "interest_rate": 0.045}
```

---

## 🔄 **How It Fits With Pandas as a Service**

```
Natural Language Request
           ↓
Pandas as a Service (SPC Generator)
           ↓
    declarative-py (Logic Engine) 
           ↓
   Production System
```

**Pandas as a Service** = Visual pipeline design + AI co-pilot  
**declarative-py** = Business logic execution + validation + rules

---

## 🚀 **Get Started**

```bash
pip install declarative-py
```

```python
from declarative_py import validate, RuleEngine

# Validate data
is_valid, errors = validate({"name": "Alice", "age": "25"}, 
                           {"name": "str", "age": "int"})

# Run business rules  
engine = RuleEngine("""
rules:
  - if: "age >= 18"
    then: {"status": "adult"}
  - else: 
    then: {"status": "minor"}
""")

result = engine.run({"age": 25})  # {"status": "adult"}
```

---

## 💰 **The Bottom Line**

**Stop writing framework code. Start writing business logic.**

`declarative-py` gives you the power of modern Python ecosystems without the complexity. Write less code, ship faster, and let AI handle the edge cases.

**Business logic should be declarative. Now it is.**

---

*`declarative-py` - Because your time is better spent on insights, not infrastructure.*
