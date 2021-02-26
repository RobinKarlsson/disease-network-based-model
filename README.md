Python 3.8

Required libraries: numpy, matplotlib

# disease-network-based-model
 network based model for the spread of an arbitrary disease in a population

# Implementation

The network is implemented as a graph where vertices represent hosts and edges represent
daily connections between hosts. Hence the neighbours of a vertex represent hosts the
vertex "meet", and forms the main avenue for infectious spread.

Hosts in the model will have four states representing their status during the outbreak. These
states will be susceptible, where a host may be infected, exposed, host has been infected
but has not yet become infectious, infected, infected and infectious, and lastly, recovered
where the host has been infected and are now immune from infection.

Vertices will add new neighbours and remove edges at predefined rates, to simulate a
dynamic population where hosts form and drop relationships over time. These new edges
may either connect a vertex to a neighbour of a neighbour or a random vertex in the population, depending on the rate of these additional edges there will, over time, be a tendency of clustering in the network, with a few connections between clusters as well as vertices
with a tendency to maintain low and high degrees.

By continuously adding and removing vertices immigration and emmigration can be simulated. Hence by having a chance that one of these new vertices are either infected or exposed recurring outbreaks of a pandemic may be simulated.

Vaccination schemes such as acquaintance vaccination and random vaccination are simulated to allow for the efficiency of different schemes to be evaluated based on netword parameters.

# default parameters

The degree of a given host is chosen as a random integer between 1 and 11, these neighbours are chosen at random from the pool of available hosts with degree less than 11, hence the initial mean degree of the hosts can be estimated as 5.5. Furthermore, as the simulation runs, each host has a probability of connecting with a neighbours neighbour at each timestep given as a normal distribution of 4.2%, and probability of dropping a connection as a normal distribution of 6.5%. The variance in both these cases are 5% of the value.

Furthermore the hosts has a chance of connecting with a random other host as a random distribution of 2% per timestep, with a variance of 8‰ from the value.

Each infectious host is, at each timestep, going to meet with a number of its neighbours.
As we have no information on the meeting habits of the population we will consider this an
arbitrary random event and model the number of neighbours a host meet at a timestep as
an exponential distribution of a third of the vertex degree.

Furthermore at each of these contacts an infectious host has a chance of spreading the
disease to its neighbour. This risk of infectious contact is modeled as a normal distribution
with 10% probability of infectious contact if the host is asymptomatic or 24% probability
of infection if the host has symptoms. The distributions will use a variance of 15% of the
probability of infectious contact. The probability of developing symptoms per timestep is
modeled as a normal distribution of 20%, with a variance of 10% of the value.

By default no vaccination schemes are in effect.
