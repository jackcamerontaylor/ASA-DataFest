println("test")
# Load packages
using Parameters
using QuantEcon
using Plots
using Format
using Interpolations
using Optim

# Define the utility function
function u(c, l, p)
    @unpack β, σ, ν, ϕ = p
    return (c^(1-σ))/(1 - σ) - ϕ * (l^(1+1/ν)/(1 + 1/ν))
end

function update_bellman!(p, V, policy, kgrid, V0, max_idxs)
    @unpack β, δ, σ, ν, α, A, ϕ = p
    n = length(kgrid)
    l_values = LinRange(0, 1, n)  # Adjusting the number of choices for labor supply
    
    vmax = -Inf
    max_idx = 0
    
    for i in 1:n
        k = kgrid[i]

        ki′ = 0
        li′ = 0
        ci′ = 0
        
        for j in 1:n
            k′ = kgrid[j]
            for b in 1:n  # Using num_l_values for the loop
                l = l_values[b]
                c′ = A * k^α * l^(1-α) + (1-δ) * k - k′
                if c′ >= 0
                    v = u(c′, l, p) + β * V0[j]
                    if v >= vmax
                        vmax = v
                        ki′ = kgrid[j]
                        li′ = l_values[b]
                        ci′ = c′
                        max_idx = i  # Store the index where vmax occurs
                    end
                end
            end
        end

        V[i] = vmax 
        policy[i] = (ki′, li′, ci′)  
    end
    
    # Store the index of vmax
    push!(max_idxs, max_idx)
end

# Implement the VFI algorithm
function solve_vfi(p, kgrid; tol = 1e-6, max_iter = 1000)
    V0 = zeros(length(kgrid))
    V = similar(V0)
    policy = Array{Tuple{Float64, Float64, Float64}, 1}(undef, length(kgrid))
    max_idxs = []  # Array to store the index of vmax
    
    for iter in 1:max_iter
        update_bellman!(p, V, policy, kgrid, V0, max_idxs)
        error = maximum(abs.(V .- V0))
        if error < tol
            println("Converged after $iter iterations.")
            return V, policy, max_idxs
        end
        copy!(V0, V)
    end
    
    error("Did not converge after $max_iter iterations.")
end

# Parameters
p = (
    β = 0.95,    # Discount Rate
    δ = 0.05,    # Depreciation of capital
    σ = 2.0,     # Uncertainty aversion
    ν = 2.0,     # Risk aversion
    α = 1/3,     # Returns to scale
    A = 1.0,     # Productivity
    ϕ = 1.0     # Disutility from labor

)

n_values = [25, 50, 75, 100]
plot_titles = ["n = 25", "n = 50", "n = 75", "n = 100"]
plots_investment = []
plots_labour = []
plots_consumption = []

for (i, n) in enumerate(n_values)
    # Define the grid for capital
    kgrid = LinRange(1e-4, 10, n)
    
    # Solve the model
    V, policy, max_idxs = solve_vfi(p, kgrid)
    
    # Extract policy functions
    ki_policy = [p[1] for p in policy]
    li_policy = [p[2] for p in policy]
    ci_policy = [p[3] for p in policy]

    # Get x and y coordinates of maximum values
    max_x_values = kgrid[max_idxs[1]]
    max_y_values = V[max_idxs[1]]
    
    # Plot policy functions with markers at maxima
    p1 = plot(kgrid, ki_policy, xlabel="Capital (k)", ylabel="Investment (k')", label="Investment Policy", title=plot_titles[i])
    push!(plots_investment, p1)
    
    p2 = plot(kgrid, li_policy, xlabel="Capital (k)", ylabel="Labor Supply (l)", label="Labor Supply Policy", title=plot_titles[i])
    push!(plots_labour, p2)
    
    p3 = plot(kgrid, ci_policy, xlabel="Capital (k)", ylabel="Consumption (c)", label="Consumption Policy", title=plot_titles[i])
    push!(plots_consumption, p3)

end
V_store_for_later, policy, max_idxs = solve_vfi(p, LinRange(1e-4, 10, 100))
display(plot(plots_investment..., layout=(2, 2), size=(800, 600)))
display(plot(plots_labour..., layout=(2, 2), size=(800, 600)))
#display(plot(plots_consumption..., layout=(2, 2), size=(800, 600)))
