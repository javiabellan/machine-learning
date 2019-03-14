# Error functions

The error function, also known as **loss** function or **cost** function,

## Single output

### Squared error

```python
def SE(target, output):
  return (target - output)**2
```

### Mean squared error

```python
def MSE():
	totalError = 0
	for i in numSamples):
		input  = data[i, 0]
		target = data[i, 1]
		output = myFunction(input)
		totalError += MSE(target, output)
	return totalError / numSamples
```

## Vector output

Imagine this data

|       | output          |  target   |
| ----- | --------------- | --------- |
| **1** | [0.1, 0.3, 0.6] | [0, 0, 1] |
| **2** | [0.2, 0.6, 0.2] | [0, 1, 0] |
| **3** | [0.3, 0.4, 0.3] | [1, 0, 0] |



### Cross entropy error

```python
# Cross entropy error
def CEE(target, output):
	error = 0
	for j in sizeOutput:
		error += ln(output)*target
	return -error
```

### Average cross entropy error

```python
def ACEE():
	totalError = 0
	for i in numSamples:
		input  = data[i, 0]
		target = data[i, 1]
		output = myFunction(input)
		totalError += CEE(target, output)
	
	return totalError / numSamples
```


