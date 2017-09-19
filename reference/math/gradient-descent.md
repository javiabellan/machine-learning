# Gradient descent recipe

Gradient descent is minimizing the error of a function.

>Graphical interpretation:
>
>Imagine that you are a blind person between the mountains, your position (x,y axis) represents the 2 parameters of some function,
and your height (z axis) represents the error in that position. So your goal is to get as low as possible in order to minimize the error.
You cant see the lowest spot, so you decide just go downhill. You can get the global minimum (lowest valley) or get suck in a local minimum.
>
>![image](http://librimind.com/wp-content/uploads/2016/03/rosenbrock-nag-copy.png)
>
> NOTE: If the function has more than 2 parameters, you can't visualize it.


## Ingredients

 * A function to optimize
 * An error measurement
 * Same data
 
> Example: Linear regression
>
>  * A function to optimize: y = mx + n
>  * Error measurement: square error
>  * Some data: A bunch of points

## Steps

### Step 1: Detect the weigts, and give them an initial value.

Optimizing a function means to optimizing its constant parameters. Detect them. Then give them a initial value.
This is a very important step because it depends on getting a good solution or not.

> For the linear regression, `y = mx + n`, `x` is the input, so `m`
 and `n` are the coefficients.
 
The height represents the error.
