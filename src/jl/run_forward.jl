#=
Utilities for normalizing and processing data output by `generate_data.jl`.
To run, load a `julia` prompt and type `include("normalize_data.jl")`. From there, run the `normalize_data` function with appropriate arguments. 
=#

using Serialization: deserialize
using HallThruster: HallThruster as het, OrderedDict
using Statistics
using NPZ
using DelimitedFiles
using FFTW: fft, fftfreq

"""
Specified the variables which should be saved in log form.
Note that we take the natural logarithm (base-e), not the base ten logarithm.
"""
const LOG_VARS = Set([:B, :nu_e, :nu_an, :nn, :ne, :ni_1, :ni_2, :ni_3, :pe, :background_pressure_torr, :frequency])

function calc_fourier_features(sample, k = nothing)
    time = Float64.(sample[:time][:time_s])
    current = Float64.(sample[:time][:discharge_current_A])
    thrust = Float64.(sample[:time][:thrust_mN])

    # Calculate Fourier transform of second half of time series (converged region)
    # Keep only frequencies > 0 and calculate mean seperately
    M = length(time)
    time = time[M÷2+1:end]
    current = current[M÷2+1:end]
    thrust = thrust[M÷2+1:end]
    dt = time[end]-time[end-1]

    N = length(time)
    mean_current = sum(current) / N
    mean_thrust = sum(thrust) / N
    freqs = fftfreq(N, 1 / dt)[2:N÷2+1]
    ampls = 2 * fft(current)[2:N÷2+1] ./ N

    # Sort complex amplitudes and frequencies by descending signal amplitude
    inds_sorted = sortperm(-abs.(ampls))

    if k !== nothing
        inds_sorted = inds_sorted[1:k]
    end

    freqs_sorted = freqs[inds_sorted]

    # divide amplitudes by mean for normalization
    ampls_sorted = ampls[inds_sorted] ./ mean_current

    # Save frequency components to dictionary
    sample[:fourier] = OrderedDict(
        :frequency => freqs_sorted,
        :real => real.(ampls_sorted),
        :imag => imag.(ampls_sorted),
    )

    # Save time-averaged performance features
    sample[:performance] = OrderedDict(
        :discharge_current_A => mean_current,
        :thrust_N => mean_thrust,
    )

    return sample
end

function load_single_sim(file::String; kwargs...)
    return load_single_sim(deserialize(file); kwargs...)
end

"""
load_single_sim(file)

Load a single simulation from data and convert it from a dictionary to a tensor with named rows.
We also check for outliers here, particularly simulations with discharge currents and thrusts that are too large (indicating sim. divergence)
or too small (indicating the thruster shut off). 
Some of the functionality here should be included during the sampling procedure, but the present approach allows us to prune simulations without regenerating them.

Returns `nothing` if simulation is deemed to be an outlier or to be excluded.
Otherwise, returns the following:

# Outputs
- param_names: list of symbols of parameter names
- param_vec: vector of parameters
- tensor_row_names: list of axially/temporally-resolved sim. output names
- tensor: tensor containing data for above variables, laid out one quantity per row.
"""
function load_single_sim(sim_dict; include_timevarying=false)
    sim_dict = calc_fourier_features(sim_dict)

	grid = collect(sim_dict[:sim]["grid"])[2:end-1]
    resolution = length(grid)

    time = sim_dict[:time][:time_s]
    I_raw = sim_dict[:time][:discharge_current_A]
    T_raw = sim_dict[:time][:thrust_mN]

    time_itp = LinRange(5.0e-4, maximum(time), resolution)
    I_itp = het.LinearInterpolation(time, I_raw).(time_itp)
    T_itp = het.LinearInterpolation(time, T_raw).(time_itp)


    # Fix some issues in the input data.
    # 1. The thrust and current cannot be negative.
    I_MAX = 150.0   # A
    I_MIN = 1.0e-1    # A
    T_MAX = 10      # N
    T_MIN = 1.0e-3    # N

    I_itp = max.(I_itp, I_MIN)
    T_itp = max.(T_itp, T_MIN)

    I_mean = sim_dict[:performance][:discharge_current_A]
    T_mean = sim_dict[:performance][:thrust_N]

    # Throw out sims with too-high or too-low thrusts and currents
	if !(I_MIN <= I_mean <= I_MAX) || !(T_MIN <= T_mean <= T_MAX)
		return :thrust_or_current_out_of_bounds
    end

    avg = sim_dict[:sim]["frames"][1]
    prop = collect(keys(avg["neutrals"]))[1]
    neutrals = avg["neutrals"][prop]
    ions = avg["ions"][prop]

    # 2. Throw out sims with min(phi) < 0.5 * V_d or max(abs(phi)) > 1.5 * V_d
    phi = avg["potential"]
    phi_min, phi_max = extrema(phi)
    V_d = sim_dict[:params][:discharge_voltage_v]
    if phi_min > 0.5 * V_d || phi_min < -10 || phi_max > 1.5 * V_d
        return :potential_out_of_bounds
    end

    # 4. Throw out sims with too-low or too-high anomalous transport
    # This should be handled better during sampling in the future
    NU_MAX_MIN = 1.0e9    # maximum of the minimum
    NU_MIN_MAX = 1.0e6    # minimum of the maximum
    NU_MIN_MIN = 1.0e4    # minimum of the minimum
    NU_MAX_MAX = 1.0e11   # maximum of the maximum
    nu_an = avg["nu_an"]
    nu_min, nu_max = extrema(nu_an)
    if !((NU_MIN_MAX < nu_max < NU_MAX_MAX) && (NU_MIN_MIN < nu_min < NU_MAX_MIN))
        return :collision_frequency_out_of_bounds
    end

    # 5. The total collision frequency must be greater than or equal to the anomalous collision frequency
    avg["nu_an"] = max.(avg["nu_an"], avg["nu_e"])

    # 6. Throw out sims with strong shocks
    ui_1 = avg["ions"][prop][1]["u"]
    max_ui_ind = argmax(ui_1)
    has_shock = max_ui_ind < 0.75 * length(ui_1) && ui_1[max_ui_ind] > 1.5 * ui_1[end]
    if has_shock
        return :ion_velocity_shock
    end

	# 7. Throw out sims with broken electron velocities
	ue = avg["ue"]
	if maximum(abs.(ue)) >= 1_000_000
		return :electron_velocity_too_large
	end

    # TODO: get these automatically
    field_names = [
        :B,
        :nu_e,
        :nu_an,
        :nn,
        :ne,
        :ni_1,
        :ni_2,
        :ni_3,
        :ui_1,
        :ui_2,
        :ui_3,
        :ue,
        :phi,
        :E,
        :Tev,
        :pe,
        :∇pe,
    ]

    if include_timevarying
        append!(field_names, [:Id, :T])
    end

    # Lay out quantities one per row into a tensor/matrix.
    # The order corresponds to the the list of names above.
    tensor_rows = Vector{Float32}[]
    for row in field_names
        row_str = String(row)
        vec = if row == :Id
            I_itp
        elseif row == :T
            T_itp
        elseif row == :phi
            avg["potential"]
        elseif row == :∇pe
            avg["grad_pe"]
        elseif row == :nn
            neutrals["n"]
        elseif startswith(row_str, "ni_")
            Z = parse(Int, row_str[4:end])
            ions[Z]["n"]
        elseif startswith(row_str, "ui_")
            Z = parse(Int, row_str[4:end])
            ions[Z]["u"]
        else
            avg[row_str]
        end

        push!(tensor_rows, vec)
    end

    # Concatenate array of arrays into a matrix
    field_tensor = hcat(tensor_rows...)

    # TODO: get these automatically
    param_names = [
        :anode_mass_flow_rate_kg_s,
        :discharge_voltage_v,
        :magnetic_field_scale,
        :cathode_coupling_voltage_v,
        :neutral_velocity_m_s,
        :wall_loss_scale,
    ]

    param_vec = [sim_dict[:params][param] for param in param_names]

    fourier_data = sim_dict[:fourier]
    fourier_names = collect(keys(fourier_data))
    fourier_tensor = hcat([fourier_data[n] for n in fourier_names]...)

    performance_data = sim_dict[:performance]
    performance_names = collect(keys(performance_data))
    performance_vec = [performance_data[n] for n in performance_names]

    time_data = sim_dict[:time]
    time_names = collect(keys(time_data))
    time_tens = hcat([time_data[n] for n in time_names]...)

    return (;
        params = (param_names, param_vec),
        fields = (field_names, field_tensor),
        fourier = (fourier_names, fourier_tensor),
        performance = (performance_names, performance_vec),
        time = (time_names, time_tens),
        grid = grid
    )
end

"""
save_sim(sim, params)

Save a simulation to a dictionary after averaging it in time for the specified interval.
In addition to axially-resolved fields, we also write out certain time-dependent global quantities (thrust, current)
as well as the params with which the simulation was run.
"""
function save_sim(sim, params = nothing)
    avg = if length(sim.frames) > 1
        het.time_average(sim, sim.t[end] / 2)
    else
        sim
    end

    out_dict = Dict(
        :sim => het.serialize(avg),
        :time => Dict(
            :time_s => sim.t .|> Float32,
            :discharge_current_A => het.discharge_current(sim) .|> Float32,
            :thrust_mN => het.thrust(sim) .|> Float32,
        ),
        :params => params,
    )

    return out_dict
end

#===============
# Main loop
===============#

input_dir = ARGS[1]
output_dir = ARGS[2]
input_files = readdir(input_dir, join=true)

Threads.@threads for i in eachindex(input_files)
    in_file = input_files[i]
    base = splitext(basename(in_file))[1]
    out_file = joinpath(output_dir, "$(base).npz")

    sol = het.run_simulation(in_file)
    if sol.retcode != :success
        continue
    end

    params = Dict(
        :anode_mass_flow_rate_kg_s => sol.config.propellants[].flow_rate_kg_s,
        :neutral_velocity_m_s => sol.config.propellants[].velocity_m_s,
        :discharge_voltage_v => sol.config.discharge_voltage,
        :cathode_coupling_voltage_v => sol.config.cathode_coupling_voltage,
        :magnetic_field_scale => sol.config.magnetic_field_scale,
        :wall_loss_scale => sol.config.wall_loss_model.loss_scale,
    )

    sim_dict = save_sim(sol, params)

    output = load_single_sim(sim_dict)

    if !(output isa Symbol)
        out_dict = Dict(
            "params" => Float32.(output.params[2]),
            "data" => Float32.(output.fields[2])',
            "fourier" => Float32.(output.fourier[2]),
            "perf" => Float32.(output.performance[2]),
            "time" => Float32.(output.time[2]),
        )
        NPZ.npzwrite(out_file, out_dict)
    end
end
