############### Cloud Computing (Revenue Management) Problem ##################
###### 1-2:
resource_req = [16 8  1;
                32 16 1;
                64 32 1;
                8  32 1;
                16 64 1;
                32 128 1;
                16 16 2;
                32 32 6;
                64 64 8];

B = [512, 1024, 64];
prices = [7, 12, 24, 22, 44, 88, 30, 90, 120];
rates = [5, 5, 1.8, 3, 2.6, 1, 0.8, 0.4, 0.3];
request = rates * 5

using JuMP, Gurobi

m = Model(solver = GurobiSolver())
@variable(m, x[1:9] >= 0)

@constraintref resource_cstr[1:3]
for j in 1:3
    resource_cstr[j] = @constraint(m, sum(x[i] * resource_req[i,j] for i in 1:9) <= B[j])
end

@constraintref request_cstr[1:9]
for i in 1:9
    request_cstr[i] = @constraint(m, x[i] <= request[i])
end

@objective(m, Max, sum(prices[i] * x[i] for i in 1:9))
status = solve(m)

revenue_obj = getobjectivevalue(m); @show revenue_obj
allocation = getvalue(x); @show allocation; @show request

# 1-2-c: more GPU?
used = allocation' * resource_req
used

getdual(resource_cstr)

# 1-2-d: Add a server
B2 = B + [32, 16, 0]

m2 = Model(solver = GurobiSolver())
@variable(m2, x[1:9] >= 0)

@constraintref resource_cstr2[1:3]
for j in 1:3
    resource_cstr[j] = @constraint(m2, sum(x[i] * resource_req[i,j] for i in 1:9) <= B2[j])
end

@constraintref request_cstr2[1:9]
for i in 1:9
    request_cstr[i] = @constraint(m2, x[i] <= request[i])
end

@objective(m2, Max, sum(prices[i] * x[i] for i in 1:9))
status = solve(m2)

revenue_obj2 = getobjectivevalue(m2); @show revenue_obj2
allocation2 = getvalue(x); @show allocation2; @show request


####### 1-3: myopic policy
# 1-3-a
using Distributions

function generateArrivalSequences( nSimulations, rates, T )
    total_rate = sum(rates);

    arrival_sequences_times = Array{Float64}[];
    arrival_sequences_types = Array{Int64}[];

    for s in 1:nSimulations
        single_arrival_sequence_time = Float64[];
        single_arrival_sequence_type = Int64[];
        t = 0;
        while (t < T)
            single_time = rand(Exponential(1/total_rate))
            single_type = rand(Categorical( rates / total_rate))

            t += single_time;

            if (t < T)
                push!(single_arrival_sequence_time, t)
                push!(single_arrival_sequence_type, single_type)
            else
                break;
            end
        end

        push!(arrival_sequences_times, single_arrival_sequence_time)
        push!(arrival_sequences_types, single_arrival_sequence_type)
    end

    return arrival_sequences_times, arrival_sequences_types
end

nSimulations = 100
types = [1,2,3,4,5,6,7,8,9]
T = 5
nResources = 3

srand(10); arrival_sequences_times = generateArrivalSequences(nSimulations, rates, T)[1]
srand(10); arrival_sequences_types = generateArrivalSequences(nSimulations, rates, T)[2]
arrival_sequences_times[1]

# 1-3-b
using StatsBase
n_count = zeros(9)
for i in 1:100
    for k in 1:9
        for j in 1:length(arrival_sequences_types[i])
            if arrival_sequences_types[i][j] == k
                n_count[k] +=  1
            end
        end
    end
end
n_count

avg_total_arrival = zeros(Int, 100, 1)
for i in 1:100
    avg_total_arrival[i] = length(arrival_sequences_types[i])
end
mean(avg_total_arrival)

# 1-3-c
results_myopic_revenue = zeros(nSimulations)
results_myopic_remaining_capacity = zeros(Int64, nResources, nSimulations)
for s in 1:nSimulations
    b = copy(B)
    single_revenue = 0.0; # will contain the revenue of this simulation
    nArrivals = length(arrival_sequences_times[s]);

    # Go through the arrivals in sequence
    for j in 1:nArrivals
        # Obtain the time of the arrival, and its type (i)
        arrival_time = arrival_sequences_times[s][j]
        i = arrival_sequences_types[s][j]

        # Check if there is sufficient capacity for the request
        if  all(b - resource_req[i, 1:3] .>= 0)
            single_revenue += prices[i]
            b -= resource_req[i, 1:3]
        end
    end
    results_myopic_revenue[s] = single_revenue
    results_myopic_remaining_capacity[1:3, s] = b'
end

# Display the results
@show mean(results_myopic_revenue)
@show mean(results_myopic_remaining_capacity, 2)


####### 1-4: BPC
m = Model(solver = GurobiSolver())
@variable(m, x[1:9] >= 0)

@constraintref resource_cstr[1:3]
for j in 1:3
    resource_cstr[j] = @constraint(m, sum(x[i] * resource_req[i,j] for i in 1:9) <= B[j])
end

@constraintref request_cstr[1:9]
for i in 1:9
    request_cstr[i] = @constraint(m, x[i] <= request[i])
end

@objective(m, Max, sum(prices[i] * x[i] for i in 1:9))

nInstances = 9
function bpc(b, t)
    for j in 1:nResources
        # Set the RHS of the resource constraint to b[r] here
        JuMP.setRHS(resource_cstr[j], b[j])
    end
    for i in 1:nInstances
        # Set the RHS of the forecast constraint for each instance
        # type to the expected number of requests over the duration
        # of the remaining horizon (T - t).
        JuMP.setRHS(request_cstr[i], (T-t) * rates[i] )
    end
    solve(m)
    # Obtain the dual values/shadow prices
    dual_val = getdual(resource_cstr)
    # Return the dual values:
    return dual_val
end


results_revenue = zeros(nSimulations)
results_remaining_capacity = zeros(Int64, nResources, nSimulations)
for s in 1:nSimulations
    b = copy(B); #Initialize the current capacity
    single_revenue = 0.0; #Initialize the revenue garnered in this simulation
    nArrivals = length(arrival_sequences_times[s]);
    for j in 1:nArrivals
        # Take the next arrival time and type from the sequence
        arrival_time = arrival_sequences_times[s][j]
        i = arrival_sequences_types[s][j]
        # Check if there is enough capacity
        if (all(b - resource_req[i,:] .>= 0))
            # Re-solve the LO and obtain the dual values
            dual_val = bpc(b, arrival_time)
            # Check if the revenue is at least the sum of the bid prices:
            if ( prices[i] >= sum(dual_val .* resource_req[i,:]))
                # If there is sufficient capacity, accrue the revenue
                # and remove the capacity.
                single_revenue += prices[i]
                b -= resource_req[i, :]
            end
        end
    end
    # Save the results of this simulation here:
    results_revenue[s] = single_revenue
    results_remaining_capacity[:, s] = b
end

@show mean(results_revenue)
@show mean(results_remaining_capacity, 2)


##########################################################
################ Designing a Sushi Menu ##################

####### 2-1:
sushi_utilities = readcsv("sushi_utilities_mat.csv")
sushi_info = readcsv("sushi_info.csv")
# 2-1-a
cust_1_most = sortperm(sushi_utilities[:,1], rev=true)[1:5]
sushi_info[cust_1_most, 1]

# 2-1-b
cust_2_least = sortperm(sushi_utilities[:,2] )[1:5]
sushi_info[cust_2_least, 1]

# 2-1-c
using StatsBase
sushi_rank = zeros(Int, 100, 500)
for i in 1:500
    sushi_rank[:,i] = ordinalrank(sushi_utilities[:,i], rev = true)
end
sushi_rank
avg_sushi_rank = mean(sushi_rank, 2 )
top_5 = sortperm(reshape(avg_sushi_rank,100))[1:5]
sushi_info[top_5, 1:2]

# 2-1-d
worst = sortperm(reshape(avg_sushi_rank,100), rev = true)[1]
sushi_info[worst, 1]

# 2-1-e
sushi_stdv = reshape(std(sushi_rank, 2 ),100)
sushi_info[sortperm(sushi_stdv, rev =true)[1], 1]


####### 2-2:
K= 500
n = 100
nopurchase_utilities = ones(Int, 1, 500) .* 3
nopurchase_info = ["no_purchase"  "n"   0]

new_sushi_utilities = cat(1, sushi_utilities, nopurchase_utilities)
new_sushi_info = cat(1, sushi_info, nopurchase_info)

function revenue_fn(S, generic_vec)
    obj = 0.0;
    for k in 1:K
        choice = S[indmax(new_sushi_utilities[S,k])];
        obj +=  1/K * generic_vec[choice]
    end
    return obj
end

# test function
@show revenue_fn([1,2,3,4,5,101], new_sushi_info[:,3])

# 2-2-a
@show revenue_fn(1:101, new_sushi_info[:,3])

# 2-2-b
top_revenue = sortperm(new_sushi_info[:,3], rev = true )[1:10]
sushi_info[top_revenue,1 ]
@show revenue_fn( [top_revenue; 101] , new_sushi_info[:,3])

# 2-2-c
cust_rank = zeros(Int, 100, 500)
for i in 1:500
    cust_rank[:,i] = sortperm(sushi_utilities[:,i], rev = true)
end

most_prefered = unique(cust_rank[1,:])
@show revenue_fn( [most_prefered; 101] , new_sushi_info[:,3])


####### 2-3:
# 2-3-b
m_sushi_1 = Model(solver = GurobiSolver())
@variable(m_sushi_1, x[1:n] >= 0 )
@variable(m_sushi_1, y[1:K, 1:(n+1)] >= 0 )

for k in 1:K
    @constraint(m_sushi_1, sum( y[k,i] for i in 1:(n+1)) == 1)

    for i in 1:n
        @constraint(m_sushi_1, y[k,i] <= x[i])
    end
    for i in 1:n
        @constraint(m_sushi_1, sum(new_sushi_utilities[i2, k] * y[k, i2] for i2 in 1:(n+1)) >= new_sushi_utilities[i,k] * x[i] + new_sushi_utilities[n+1, k] * (1-x[i]) )
    end
    @constraint(m_sushi_1, sum(new_sushi_utilities[i2,k] * y[k, i2] for i2 in 1:(n+1)) >= new_sushi_utilities[n+1,k] )
end

@constraint(m_sushi_1, x[1:n] .<= 1)
@constraint(m_sushi_1, y[1:K, 1:(n+1)] .<= 1)

@objective(m_sushi_1, Max, sum( 1/K * new_sushi_info[i,3] * y[k, i] for k in 1:K, i in 1:n))
solve(m_sushi_1)

obj_sushi = getobjectivevalue(m_sushi_1); @show obj_sushi

# 2-3-d
m_sushi_2 = Model(solver = GurobiSolver())
@variable(m_sushi_2, x[1:n] >= 0, Bin)
@variable(m_sushi_2, y[1:K, 1:(n+1)] >= 0 , Bin)

for k in 1:K
    @constraint(m_sushi_2, sum( y[k,i] for i in 1:(n+1)) == 1)

    for i in 1:n
        @constraint(m_sushi_2, y[k,i] <= x[i])
    end
    for i in 1:n
        @constraint(m_sushi_2, sum(new_sushi_utilities[i2, k] * y[k, i2] for i2 in 1:(n+1)) >= new_sushi_utilities[i,k] * x[i] + new_sushi_utilities[n+1, k] * (1-x[i]) )
    end
    @constraint(m_sushi_2, sum(new_sushi_utilities[i2,k] * y[k, i2] for i2 in 1:(n+1)) >= new_sushi_utilities[n+1,k] )
end


@objective(m_sushi_2, Max, sum( 1/K * new_sushi_info[i,3] * y[k, i] for k in 1:K, i in 1:n))
solve(m_sushi_2)

obj_sushi2 = getobjectivevalue(m_sushi_2); @show obj_sushi2

@show getvalue(x)
optimal_sushi_unconstrained2 = find(getvalue(x) .> 0.5)
new_sushi_info[optimal_sushi_unconstrained2,1]

@show revenue_fn([optimal_sushi_unconstrained2; 101], new_sushi_info[:,3] ) # double check

####### 2-4

m_sushi_3 = Model(solver = GurobiSolver())
@variable(m_sushi_3, x[1:n] >= 0, Bin)
@variable(m_sushi_3, y[1:K, 1:(n+1)] >= 0 , Bin)

for c in 0:11
    C = (sushi_info[:,2] .== c)
    @constraint(m_sushi_3, sum(x[1:n] .* C) >= 1)
end

for k in 1:K
    @constraint(m_sushi_3, sum( y[k,i] for i in 1:(n+1)) == 1)

    for i in 1:n
        @constraint(m_sushi_3, y[k,i] <= x[i])
    end
    for i in 1:n
        @constraint(m_sushi_3, sum(new_sushi_utilities[i2, k] * y[k, i2] for i2 in 1:(n+1)) >= new_sushi_utilities[i,k] * x[i] + new_sushi_utilities[n+1, k] * (1-x[i]) )
    end
    @constraint(m_sushi_3, sum(new_sushi_utilities[i2,k] * y[k, i2] for i2 in 1:(n+1)) >= new_sushi_utilities[n+1,k] )
end

@objective(m_sushi_3, Max, sum( 1/K * new_sushi_info[i,3] * y[k, i] for k in 1:K, i in 1:n))
solve(m_sushi_3)

obj_sushi3 = getobjectivevalue(m_sushi_3); @show obj_sushi3

@show getvalue(x)
optimal_sushi_unconstrained3 = find(getvalue(x) .> 0.5)
new_sushi_info[optimal_sushi_unconstrained3,1]
@show revenue_fn([optimal_sushi_unconstrained3; 101], new_sushi_info[:,3] ) # double check
