#Blotor
using Plots, Distributions, Statistics, Images, StatsBase





# function randwalk(A,stepno=5000,inits=10)
#     indis = vec(CartesianIndices(A))
#     for j in 1:inits
#         p_0 = indis[rand(1:length(indis))]
#         A[p_0] = 1
#         for i in 1:stepno
#             step = CartesianIndex(rand(-1:1),rand(-1:1))
#             p_0 += step
#             if p_0[1] <= 0
#                 p_0 =  CartesianIndex(size(A)[1],p_0[2])

#             end
#             if p_0[2] <= 0
#                 p_0 = CartesianIndex(p_0[1],size(A)[2])
#             end
#             if p_0[1] > size(A)[1]
#                 p_0 = CartesianIndex(1,p_0[2])
#             end
#             if p_0[2] > size(A)[2]
#                 p_0 = CartesianIndex(p_0[1],1)
#             end
#             A[p_0] = 1
#         end
#     end
# return A
# end

# function gaussblots(A,centres,samples)
#     indis = vec(CartesianIndices(A))
#     for i in 1:centres
#         centre = indis[rand(1:length(indis))]
#         μ1 = centre[1]
#         μ2 = centre[2]
#         σ1 = rand(1:size(A)[1]/7)
#         σ2 =  rand(1:size(A)[2]/7)
#         p1 = Int.(round.(rand(Truncated(Normal(μ1,σ1),1,size(A)[1]),samples)))
#         p2 = Int.(round.(rand(Truncated(Normal(μ2,σ2),1,size(A)[1]),samples)))
#         I = CartesianIndex.(p1,p2)
#         A[I] .= 1
#     end
#     return A
# end

function eulrsmooth(A,α=1)
    indis = findall(==(1),A)
    allindis = CartesianIndices(A)
    for i in indis
        for j in allindis
                A[j] += exp(-α*sqrt((i[1]-j[1])^2+(i[2]-j[2])^2))
        end
        
        
    end
    return A
end


function getCentres(A,m=3,mutate=false) 
    indis = vec(CartesianIndices(A))
    r = rand(length(indis))
    tresh = 1 - (m/length(indis))
    c = findall(>(tresh),r)
    if mutate
        return A[c]
    end
    return indis[c]

end

softmax(x) = exp.(x)./(sum(exp.(x)))
ratio(x) = x./(sum(x))


function genP(A,temp=10)
    P = []
    indis = vec(CartesianIndices(A))
    for i in indis
        push!(P,AnalyticWeights(softmax(rand(8))))
    end
    return P 

end

function ParamedRW(A,P,centres,stepno=1000)
    Po = copy(P)
    indis = vec(CartesianIndices(A))
    moves = [CartesianIndex(1,1),CartesianIndex(0,1),CartesianIndex(1,0),CartesianIndex(0,-1),CartesianIndex(-1,0),CartesianIndex(-1,1),CartesianIndex(1,-1),CartesianIndex(-1,-1)]
    for i in centres
        p = i
        A[p] += 1
        for j in 1:stepno  
            move = sample(moves,P[findall(==(p),indis)[1]])
            mul = ones(8)
            mul[findall(==(move),moves)[1]] = ℯ^2
            Po[findall(==(p),indis)[1]] =  AnalyticWeights(softmax(P[findall(==(p),indis)[1]].*mul))   
            p += move
            if p[1] <= 0
                p =  CartesianIndex(size(A)[1],p[2])

            end
            if p[2] <= 0 
                p = CartesianIndex(p[1],size(A)[2])
            end
            if p[1] > size(A)[1]
                p = CartesianIndex(1,p[2])
            end
            if p[2] > size(A)[2]
                p = CartesianIndex(p[1],1)
            end
            A[p] = 1 
         end
    end
    return A, Po
end 


function makeBlot(dims=(200,200),P=nothing,centres=nothing)
    A = zeros(dims)
    if P === nothing
        P = genP(A)
    end
    if centres === nothing
        centres = getCentres(A)
    end
    A, P = ParamedRW(A,P,centres,length(A)/10)
    A = eulrsmooth(A,0.6)
    A  = [A[:,end:-1:1] A]
    blot = (A.-minimum(A)) ./(maximum(A)-minimum(A))
    return blot, P, centres
end



function  GenGen(size=10,Ps=nothing,Cs=nothing)
    blots = []
    Pso = []
    Cso = []
    if Ps === nothing
        for i in 1:size
            temp = makeBlot((200,200))
            push!(blots,temp[1])
            push!(Pso,temp[2])
            push!(Cso,temp[3])
          
        end
    else
        for i in 1:length(Ps)
            temp = makeBlot((200,200),Ps[i],Cs[i])
            push!(blots,temp[1])
            push!(Pso,temp[2])
            push!(Cso,temp[3])
        end
    end
    return blots, Pso, Cso
end





function mutate(Ps,Cs,outsize=6)
    Pso = []
    Cso = []
    for i in 1:outsize
        combo = softmax.(rand(length(Ps)))
        push!(Pso,AnalyticWeights.(sum(combo.*Ps)))
        s = 0
        for i in Cs
            s += length(i)
        end
        pool = collect(Iterators.flatten(Cs))
        push!(Cso,getCentres(pool,div(s,length(Cs)),true))
    end
    return Pso, Cso
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))


function  Blotor(popsize=10)
    Ps = nothing
    Cs = nothing
    while true
        blots, Ps, Cs = GenGen(popsize,Ps,Cs)
        if length(blots)%2 != 0 
            push!(blots,zeros(size(blots[1])))
        end
        a = blots[1]
        b = blots[end]
        for i in 2:Int(length(blots)/2)
            a = hcat(a,blots[i])
            b = hcat(b,blots[length(blots)+1-i])
        end
        G = vcat(a,b)
        print("Displaying current generation:")
        display(colorview(Gray,map(clamp01nan,G)))
        print("Choose progenitors"*"\n"*"Please input like: 1 5 7"*"\n"*"type q to quit"*"\n\n"*":")
        progens = readline()
        if progens == "q"
            return blots
        end
        progens =  vec([parse(Int, x) for x in split(progens)])
        Ps, Cs = mutate(Ps[progens],Cs[progens],popsize)
    end

end

 