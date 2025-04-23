
import irispie as ir
import functools as ft

start_period = ir.qq(2025,1)
num_periods = 20
span = start_period >> start_period + num_periods - 1
printn = ft.partial(print, end="\n\n", )


m = ir.Simultaneous.from_pickle_file("model.pkl", )

#
# Create a Stacker object for a specific Simultaneous model and a specific time
# span. Optionally, select only a subset of transition and measurement
# variables, and anticipated and unanticipated shocks (remember,  anticipated
# shocks are deterministic).
#
# Note that if any of these arguments are `None`, **all** the respective
# variables or shocks are used.
#

selected_transition_variables = ("y_gap", "rs", "cpi", "ad_cpi", )
selected_measurement_variables = ("obs_y", "obs_cpi", "obs_rs", )

selected_unanticipated_shocks = m.get_names(kind=ir.UNANTICIPATED_SHOCK, )
# equivalent to:
# selected_unanticipated_shocks = None

t = ir.Stacker.from_simultaneous(
    m, span,
    transition_variables=selected_transition_variables,
    measurement_variables=selected_measurement_variables,
    unanticipated_shocks=selected_unanticipated_shocks,
    anticipated_shocks=None,
    measurement_shocks=None,
)

vec = t.stacked_vector
names = t.transition_variable_names
periods = t.base_periods

# Vector of transition variable names included in the stacker
printn(names)

# Vector of dated transition variables included in the stacker
printn(vec)

# Vector of transition variable names included in the stacker
printn(names)

# Vector of periods included in the stacker
printn(periods)


#
# Set up a Databox with measurement variables upon which the marginal
# distribution will be conditioned. Only the measurement variables selected when
# creating the Stacker object will be considered. Their input time series may
# include missing observations.
#

db = ir.Databox.steady(m, span, )
db.keep(selected_measurement_variables, )

db["obs_rs"][start_period] = 5
db["obs_y"][start_period+2>>ir.end>>3] = None


#
# Calculate the marginal distribution of the selected transition variables. The
# marginal distribution is described by its mean and covariance matrix.
#

marg_dist = t.calculate_marginal(db, stds_from_data=True, )

marg_mean, marg_mse = marg_dist

sx = marg_mean.reshape((len(names), -1), order="F", )
sx_std = ir.std_from_cov(marg_mse).reshape((len(names), -1), order="F", )


#
# Create a CovarianceSimulator object for resampling from the marginal
# distribution. The shape of the result is the length of the stacked vector of
# the transition variables times the number of periods by the number of draws
# requested.
#

cs = ir.CovarianceSimulator(cov=marg_mse, mean=marg_mean, )

C = cs.factor

ksi = cs.simulate(num_draws=100_000, )
print(ksi.shape, )


