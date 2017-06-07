from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

f_val0 = f()
f_val1 = f()

# print f_val1

g_val0 = g()
g_val1 = g()

# print g_val0
# print g_val1

rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
rng_val.seed(902340)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng

f = function([], rv_u)
f_val2 = f()

# print f_val2

state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()       # this affects rv_u's generator

v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()             # v2 != v1
v3 = f()             # v3 == v1

print v1
print v2
print v3
