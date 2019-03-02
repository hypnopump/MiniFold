"""
	This is a script alternative to the julia_get_proteins_under_200_aa.ipynb
	notebook for those who don't have IJulia or would like to run it as a script

	Notebook to preprocess the raw data file and
	handle it properly. 
	Will prune the unnecessary data for now.
	Reducing data file from 600mb to 170mb.

	Select only proteins under L aminoacids (AAs).
"""

L = 200										# Set maximum AA length
N = 995										# Set maximum number of proteins
RAW_DATA_PATH = "../data/training_30.txt"	# Path to raw data file
DESTIN_PATH = "../data/full_under_200.txts"	# Path to destin file

# Open the file and read content

f = try open(RAW_DATA_PATH) catch
	println("File not found. Check it's there. Instructions in the readme.")
	exit(0)
	end
lines = readlines(f)



function coords_split(lister, splice)
    # Split all passed sequences by "splice" and return an array of them
    # Convert string fragments to float 
    coords = []
    for c in lister
        push!(coords, [parse(Float64, a) for a in split(c, splice)])
    end
    return coords
end


function norm(vector)
	# Could use "Using LinearAlgebra + built-in norm()" but gotta learn Julia
    return sqrt(sum([v*v for v in vector]))
end


# Scan first n proteins
names = []
seqs = []
coords = []
pssms = []

try
    # Record names, seqs and coords for each protein btwn 1-n
    for i in 1:length(lines)
        if length(coords) == N
            break
        end
        
        # Start recording
        if lines[i] == "[ID]"
            push!(names, lines[i+1]) 
        elseif lines[i] == "[PRIMARY]"
            push!(seqs, lines[i+1])
        elseif lines[i] == "[TERTIARY]"
            push!(coords, coords_split(lines[i+1:i+3], "\t"))
        elseif lines[i] == "[EVOLUTIONARY]"
            push!(pssms, coords_split(lines[i+1:i+21], "\t"))
            # Progress control
            if length(names)%50 == 0
                println("Currently @ ", length(names), " out of n: ", N)
            end
        end  
    end
catch
    println("Error while reading file. Check it's complete or download again.")
    exit(0)
end


# Check proteins w/ length under L
println("\n\nTotal number of proteins: ", length(seqs))
under = []
for i in 1:length(seqs)
    if length(seqs[i])<L
        push!(under, i)
        # Uncomment for debugging purposes
        # println("Seelected with: ", length(seqs[i]), " number: ", i)
    end
end
println("Number of proteins under ", L, " : ", length(under), "\n\n")


# Get distances btwn pairs of AAs - only for prots under 200
dists = []
try
    for k in under
        # Get distances from coordinates
        dist = []
        for i in 1:length(coords[k][1])
            # Only pick coords for C-alpha carbons! - position (1/3 of total data)
            # i%3 == 2 Because juia arrays start at 1 - Python: i%3 == 1
            if i%3 == 2
                aad = [] # Distance to every AA from a given AA
                for j in 1:length(coords[k][1])
                    if j%3 == 2
                        push!(aad, norm([coords[k][1][i],coords[k][2][i],coords[k][3][i]]
                    	    			- 
                    		    		[coords[k][1][j],coords[k][2][j],coords[k][3][j]]))
                    end
                end
                push!(dist, aad)
            end
        end
        push!(dists, dist)
    
        # Progress control
        if length(dists)%50 == 0
            println("Dists Currently @ ", length(dists), " out of n: ", N)
        end
    end
catch
	println("Error while calculating distances. Set N to smaller value.")
end


# Check everything's alright
n = 2
println("\n\nSample protein data (example)")
println("id: ", names[n])
println("seq: ", seqs[n]) 
println("sample coord: ", coords[n][1][1]) 
println("sample dist: ", dists[n][1][5])


# Data is OK. Save it to a file.
using DelimitedFiles
open(DESTIN_PATH, "w") do f
    aux = [0]
    for k in under
        push!(aux, aux[length(aux)]+1)
        # ID
        write(f, "\n[ID]\n")
        write(f, names[k])
        # Seq
        write(f, "\n[PRIMARY]\n")
        write(f, seqs[k])
        # PSSMS
        write(f, "\n[EVOLUTIONARY]\n")
        writedlm(f, pssms[k])
        # Coords
        write(f, "\n[TERTIARY]\n")
        writedlm(f, coords[k])
        # Dists
        write(f, "\n[DIST]\n")
        # Check that saved proteins are less than 200 AAs
        if length(dists[aux[length(aux)]][1])>L
            println("error when checking protein in dists n: ",
            		 aux[length(aux)], " length: ", length(dists[aux[length(aux)]][1]))
            break
        else
            writedlm(f, dists[aux[length(aux)]])
        end
    end
end


println("\n\nScript execution went fine. Data is ready at: ", DESTIN_PATH)
exit(0)