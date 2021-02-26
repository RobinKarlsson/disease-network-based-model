import random
import numpy as np
import matplotlib.pyplot as plt

'''
Vertex or agent/host object

input:
    int time - timestep the object was created, resets any time the objects status changes to reflect timestep of change
    str status - whether the objects initiates as susceptible, exposed, infected or recovered
    list neighbours - neighbours of vertex, list contain Vertex objects
    int gamma - average timesteps spent in infected stage before becoming recovered
    float beta - probability of infecting a selected neighbour each timestep if vertex doesnt have symptoms
    float beta_symptoms - probability of infecting a selected neighbour each timestep if vertex has symptoms
    float prob_symptoms - probability of developing syptoms per timestep while infectious
    float prob_death - probability of dying per timestep during infection if vertex doesnt have symptoms
    float prob_death_symptoms - probability of dying per timestep during infection if vertex has symptoms
    int immunity_duration_min - min timesteps spent in recovered state before becoming susceptible
    int immunity_duration_max - max timesteps spent in recovered state before becoming susceptible
    float new_neighbour_rate - probability of forming edge to a random neigbours neighbour per timestep
    float drop_neighbour_rate - probability of vertex degree decreasing by 1 per timestep
    int latency - average timesteps spent in exposed stage before becoming infected
    float exp_divide - divides number of contacts in the exponential distribution. Increased exp_divide reduces the number of meetings/timestep
'''
class Vertex():
    def __init__(self, time, status = 'susceptible', neighbours = [], gamma = 7, beta = 0.1, beta_symptoms = 0.24,
                 prob_symptoms = 0.2, prob_death = 0, prob_death_symptoms = 0, immunity_duration_min = 5000,
                 immunity_duration_max = 6000, new_neighbour_rate = 0.042, drop_neighbour_rate = 0.065, latency = 5,
                 exp_divide = 3.0):

        self.susceptible, self.exposed, self.infected, self.recovered, self.time = False, False, False, False, time
        self.alive = True

        #probability that infected vertex without symptoms infect neighbour, with symptoms
        self.beta, self.beta_symptoms = abs(np.random.normal(beta, beta*0.15)), abs(np.random.normal(beta_symptoms, beta_symptoms*0.15))

        #time before recovering, time spent in exposed state before becoming infectious as a normal distribution
        self.gamma, self.latency = abs(np.random.normal(gamma)), abs(np.random.normal(latency))

        #probability of infected vertex dying without symptoms, probability of death with symptoms
        self.prob_death, self.prob_death_symptoms = prob_death, prob_death_symptoms

        #probability of symptom onset
        self.prob_symptoms = abs(np.random.normal(prob_symptoms, prob_symptoms * 0.1))

        #time vertex is immune after recovering before becoming susceptible again
        self.immunity_duration = random.randint(immunity_duration_min, immunity_duration_max)

        #probability of edge forming between vertex and a neighbours neighbour, probability vertex loses an edge
        self.new_neighbour_rate = abs(np.random.normal(new_neighbour_rate, new_neighbour_rate * 0.05))
        self.drop_neighbour_rate = abs(np.random.normal(drop_neighbour_rate, drop_neighbour_rate * 0.05))

        self.exp_divide = exp_divide

        #list of neighbours of vertex
        self.neighbours = [v for v in neighbours] 

        self.has_symptoms = False
        if(status == 'susceptible'):
            self.susceptible = True
        elif(status == 'exposed'):
            self.exposed = True
        elif(status == 'infected'):
            self.infected = True
        elif(status == 'recovered'):
            self.recovered = True


    '''
    Add an edge to the object, connecting it to a new neighbour
    
    Input:
        Vertex vertex - Vertex object to connect object with
    '''
    def addEdge(self, vertex):
        #if vertex isnt already a neigbour or self
        if not vertex in self.neighbours + [self]:
            self.neighbours.append(vertex)

    '''
    Update object status.
    Should be called at each timestep
    
    Input:
        int time - current timestep
        
    Output:
        bool True if object is alive, otherwise bool False
    '''
    def updateVertex(self, time):
        #form connection with random neighbours neighbour
        if(random.uniform(0, 1) < self.new_neighbour_rate): 
            try:
                #pick a random neighbour with degree > 1
                neighbour = random.choice([v for v in self.neighbours if len(v.neighbours) > 1])
                new_neighbour = random.choice([v for v in neighbour.neighbours if v != self])
                self.addEdge(new_neighbour)
                new_neighbour.addEdge(self)
            except IndexError:
                'doesnt have any valid neighbours'

        #drop a random neighbour
        if(random.uniform(0, 1) < self.drop_neighbour_rate): 
            try:
                #pick a random neighbour
                vertex = random.choice(self.neighbours)
                self.removeNeighbour(vertex)
            except IndexError:
                'doesnt have any neighbours'

        if self.exposed:
            if(self.time + self.latency <= time):
                self.setStatus('infected', time)

        if self.infected:
            prob_death = self.prob_death_symptoms if self.has_symptoms else self.prob_death
            beta = self.beta_symptoms if self.has_symptoms else self.beta

            #probability that an infected person develops symptoms
            if(random.uniform(0, 1) < self.prob_symptoms):
                self.has_symptoms = True

            #probability that an infected person dies
            if(random.uniform(0, 1) < prob_death): 
                self.alive = False

                #decreased risk of spreading on the day of death
                beta /= abs(np.random.normal(0.5, 0.1))

            #if vertex recovers on current timestep
            elif(self.time + self.gamma <= time):
                self.setStatus('recovered', time)

            #meet with a number of contacts given by an exponential distribution
            for _ in range(min(abs(round(np.random.exponential(len(self.neighbours)/self.exp_divide))), len(self.neighbours))):
                try:
                    #pick a random neigbour (might not be susceptible)
                    neighbour = random.choice(self.neighbours)

                    #if neighbour is susceptible it has a chance to get infected
                    if(neighbour.susceptible and random.uniform(0, 1) < beta):
                        neighbour.setStatus('exposed', time)

                #vertex has degree 0
                except IndexError: 
                    continue

        elif self.recovered:
            if(self.time + self.immunity_duration <= time):
                self.setStatus('susceptible', time)

        return self.alive

    '''
    Remove edges between object and another vertex
    
    input:
        Vertex vertex - Vertex object to disconnect from object
    '''
    def removeNeighbour(self, vertex):
        if(vertex in self.neighbours):
            self.neighbours.remove(vertex)
            vertex.removeNeighbour(self)

    '''
    Set status of object
    
    input:
        str status - new Status of the object
        int time - current timestep
    '''
    def setStatus(self, status, time):
        self.susceptible, self.exposed, self.infected, self.recovered = False, False, False, False
        self.has_symptoms = False

        if(status == 'susceptible'):
            self.susceptible = True
            self.time = time
        elif(status == 'exposed'):
            self.exposed = True
            self.time = time
        elif(status == 'infected'):
            self.infected = True
            self.time = time
        elif(status == 'recovered'):
            self.recovered = True
            self.time = time


















'''
Function to run simulation and return results

input:
    int timesteps - number of timesteps to run simulation for
    int population - total population of network
    int immune - number of initially immune (recovered) objects
    int min_degree - min number of neighbours of each vertex
    int max_degree - max number of neighbours of each vertex
    float prob_cold_connection - probability of vertices to connect to random vertex
    float prob_cold_connection_lonely - probability of vertices to connect to random vertex if degree is 0
    int initial_num_exposed - number of initially exposed objects
    int initial_num_infected - number of initially infectious objects
    float immigration_prob - probability of immigration happening at each timestep
    float emmigration_prob - probability of emmigration happening at each timestep
    float size_immigration - size of immigration as fraction of total population
    float size_emmigration - size of emmigration as fraction of total population
    float prob_immigrant_exposed - probability of immigrating vertex being exposed
    float prob_immigrant_infected - probability of immigrating vertex being infected
    int min_degree_immigrant - min number of neighbours of immigrating vertex
    int max_degree_immigrant - max number of neighbours of immigrating vertex
    float offspring_prob - probability two connected vertices produce a new vertex per timestep
    bool scale_immigration_emmigration - True if immigration/emmigration scales, otherwise False
    str vaccin_strategy - specify vaccination strategy if any, otherwise None
    int vaccin_doses - number of available doses vaccin
    str degree_distribution - distribution to use for degree of vertices
    list degree_data - list with data on degree distribution

output:
    list - list of vertex objects in network
    float - estimated R0
    dict - dictionary of deaths per timestep
    dict - dictionary of total population per timestep
    dict - disctionary of degree distribution of vertices
    dict - dictionary of number of susceptible vertices per timestep
    dict - dictionary of number of exposed vertices per timestep
    dict - dictionary of number of infected vertices per timestep
    dict - dictionary of number of recovered vertices per timestep
'''
def runSim(timesteps, population, immune, min_degree, max_degree, prob_cold_connection, prob_cold_connection_lonely,
           initial_num_exposed, initial_num_infected, immigration_prob, emmigration_prob, size_immigration, size_emmigration,
           prob_immigrant_exposed, prob_immigrant_infected, min_degree_immigrant, max_degree_immigrant, offspring_prob,
           scale_immigration_emmigration, vaccin_strategy, vaccin_doses, exp_divide, degree_distribution, degree_data):

    #list and dicts to store data
    vertices, fatalities, current_population, S, E, I, R = [], {}, {}, {}, {}, {}, {}

    #initiate susceptible vertices
    for _ in range(population - immune - initial_num_infected - initial_num_exposed):
        vertices.append(Vertex(time = 0, exp_divide = exp_divide))

    #initiate exposed vertices
    for _ in range(initial_num_exposed):
        vertex = Vertex(time = 0, exp_divide = exp_divide)
        vertex.setStatus('exposed', 0)
        vertices.append(vertex)

    #initiate infected vertices
    for _ in range(initial_num_infected):
        vertex = Vertex(time = 0, exp_divide = exp_divide)
        vertex.setStatus('infected', 0)
        vertices.append(vertex)

    #initiate immune vertices
    for _ in range(immune):
        vertex = Vertex(time = 0, exp_divide = exp_divide)
        vertex.setStatus('recovered', 0)
        vertices.append(vertex)

    #add edges to vertices

    #if degree distribution is random
    if(degree_distribution == 'random'):
        for vertex in vertices:
            while len(vertex.neighbours) < random.randint(min_degree, max_degree):
                candidates = [v for v in vertices if len(v.neighbours) < max_degree - 1]
                new_neighbour = random.choice(candidates)
                if(new_neighbour != vertex):
                    vertex.addEdge(new_neighbour)
                    new_neighbour.addEdge(vertex)

    #if degree distribution is a multimodal distribution
    elif(degree_distribution == 'multimodal'):
        for vertex in vertices[round(degree_data[0][1] * population): ]:
            mean = degree_data[1][0]

            while len(vertex.neighbours) < abs(round(np.random.normal(mean, mean * 0.1))):
                candidates = [v for v in vertices if len(v.neighbours) < max_degree - 1]

                new_neighbour = random.choice(candidates)

                if(new_neighbour != vertex):
                    vertex.addEdge(new_neighbour)
                    new_neighbour.addEdge(vertex)

        for vertex in vertices[0: round(degree_data[0][1] * population)]:
            mean = degree_data[0][0]

            while len(vertex.neighbours) < abs(round(np.random.normal(mean, mean * 0.1))):
                candidates = [v for v in vertices if len(v.neighbours) < max_degree - 1]

                new_neighbour = random.choice(candidates)

                if(new_neighbour != vertex):
                    vertex.addEdge(new_neighbour)
                    new_neighbour.addEdge(vertex)

    #store degree distribution
    dist = {}
    for i in range(max_degree + 2):
        dist[i] = len([v for v in vertices if len(v.neighbours) == i])
        

    #implement vaccination
    if vaccin_strategy:
        #list of vertices that has received vaccine
        vaccinated, i = [], vaccin_doses

        #acquaintance vaccination strategy
        if(vaccin_strategy == 'random'):
            while i > 0:
                vertex = random.choice([v for v in vertices if v not in vaccinated])
                vertex.setStatus('recovered', 0)
                vaccinated.append(vertex)
                i -= 1
            
        elif(vaccin_strategy == 'acquaintance_vaccination'):
            vertex = None
            
            while i > 0:
                #if no candidate for vaccin available, choose one at random
                if not vertex:
                    vertex = random.choice([v for v in vertices if v not in vaccinated])

                #vaccinate vertex
                vertex.setStatus('recovered', 0)
                vaccinated.append(vertex)

                #choose a random neighbour of vertex for vaccination
                try:
                    vertex = random.choice([v for v in vertex.neighbours if v not in vaccinated])
                    
                #no valid neighbours who can receive vaccin
                except IndexError: 
                    vertex = None
                
                i -= 1

    #estimate R0
    sum_degree, max_degree, beta, beta_symptoms, gamma, prob_symptoms = 0, 0, 0, 0, 0, 0
    for v in vertices:
        sum_degree += len(v.neighbours)
        beta += v.beta
        beta_symptoms += v.beta_symptoms
        gamma += v.gamma
        prob_symptoms += v.prob_symptoms

        if(len(v.neighbours) > max_degree):
            max_degree = len(v.neighbours)

    mean_degree = sum_degree / population
    beta /= population
    beta_symptoms /= population
    gamma /= population
    prob_symptoms /= population
    variance = max_degree - mean_degree
    p = (beta * (1 - prob_symptoms) + beta_symptoms * prob_symptoms) * gamma / exp_divide
    R0 = p * (mean_degree + (variance - mean_degree)/mean_degree)  

    #main loop, for each timestep
    for t in range(timesteps):
        #current total population, variable to store number of deaths
        current_population[t], deaths = len(vertices), 0

        #if immigration happen this timestep
        if(random.uniform(0, 1) < immigration_prob): 
            #indicated base immigration size
            base_immigration = np.random.normal(size_immigration, size_immigration * 0.005) * population 

            #scale immigration based on population size
            if scale_immigration_emmigration:
                if(current_population[t] <= population * 0.5):
                    base_immigration *= 8
                elif(current_population[t] <= population * 0.7):
                    base_immigration *= 4
                elif(current_population[t] <= population * 0.85):
                    base_immigration *= 2
                elif(current_population[t] <= population * 0.95):
                    base_immigration *= 1.5
                elif(current_population[t] < population):
                    base_immigration *= 1.2
                elif(current_population[t] > population):
                    base_immigration /= 1.2
                elif(current_population[t] > population * 1.1):
                    base_immigration /= 2
                elif(current_population[t] > population * 1.25):
                    base_immigration /= 4
                elif(current_population[t] > population * 1.5):
                    base_immigration /= 8
            
            #add indicated number of new vertices to the system
            for _ in range(int(base_immigration)):
                immigrant = Vertex(time = t, exp_divide = exp_divide)

                #if immigrant is exposed
                if(random.uniform(0, 1) < prob_immigrant_exposed):
                    immigrant.setStatus('exposed', t)

                #if immigrant is infectious
                elif(random.uniform(0, 1) < prob_immigrant_infected):
                    immigrant.setStatus('infected', t)

                #connect immigrating vertex to indicated number of neighbours
                for _ in range(random.randint(min_degree_immigrant, max_degree_immigrant)):
                    new_neighbour = random.choice([v for v in vertices if v != immigrant])
                    immigrant.addEdge(new_neighbour)
                    new_neighbour.addEdge(immigrant)

                vertices.append(immigrant)

        #if emmigration happen this timestep
        if(random.uniform(0, 1) < emmigration_prob): 
            #indicated base emmigration size
            base_emmigration = np.random.normal(size_emmigration, size_emmigration * 0.005) * population 
            
            #scale emmigration based on population size
            if scale_immigration_emmigration:
                if(current_population[t] > population):
                    base_emmigration *= 1.5
                elif(current_population[t] > population * 1.1):
                    base_emmigration *= 4
                elif(current_population[t] > population * 1.25):
                    base_emmigration *= 8
                elif(current_population[t] > population * 1.5):
                    base_emmigration *= 15
                
            #remove indicated number of vertices from system
            for _ in range(int(base_emmigration)):
                emmigrant = random.choice(vertices)

                #remove edges between emmigrating vertex and its neighbours
                while len(emmigrant.neighbours) != 0:
                    emmigrant.removeNeighbour(emmigrant.neighbours[0])

                vertices.remove(emmigrant)

        #for each vertex in the graph
        for vertex in vertices:
            #vertex doesnt have any neighbours
            if(len(vertex.neighbours) == 0): 
                #vertex looks for others to connect with
                if(random.uniform(0, 1) < prob_cold_connection_lonely): 
                    #forms edge with another random vertex, if such an vertex exist
                    if(current_population[t] > 1):
                        new_neighbour = random.choice([v for v in vertices if v != vertex])
                        vertex.addEdge(new_neighbour)
                        new_neighbour.addEdge(vertex)

            #vertex has neighbours  
            else:
                #probability of vertex producing offspring with 1 random neighbour
                if(random.uniform(0, 1) < offspring_prob): 
                    #produce offspring with two parents as its neighbours
                    parent2 = random.choice(vertex.neighbours)
                    offspring = Vertex(time = t, exp_divide = exp_divide)

                    offspring.addEdge(vertex)
                    offspring.addEdge(parent2)
                    vertex.addEdge(offspring)
                    parent2.addEdge(offspring)

            if(random.uniform(0, 1) < prob_cold_connection):
                #vertex makes a connection with another vertex without having mutual neighbours
                if(current_population[t] > 1):
                    new_neighbour = random.choice([v for v in vertices if v != vertex])
                    vertex.addEdge(new_neighbour)
                    new_neighbour.addEdge(vertex)

            #update vertex
            isAlive = vertex.updateVertex(t) 

            #vertex died from decease
            if not isAlive:
                #increment number of deaths at current timestep and disconnect & remove dead vertex
                deaths += 1
                while len(vertex.neighbours) != 0:
                    vertex.removeNeighbour(vertex.neighbours[0])

                vertices.remove(vertex)

        #count number of vertices in each stage at current timestep
        S[t], E[t], I[t], R[t] = 0, 0, 0, 0
        for v in vertices:
            if(v.susceptible):
                S[t] += 1
            elif(v.exposed):
                E[t] += 1
            elif(v.infected):
                I[t] += 1
            elif(v.recovered):
                R[t] += 1

        #vaccinated hosts are not considered recovered from disease
        if vaccin_strategy:
            R[t] -= vaccin_doses

        #number of deaths at current timestep
        fatalities[t] = deaths

    return vertices, R0, fatalities, current_population, dist, S, E, I, R













if __name__ == '__main__':
    #timesteps to run simulation for, initial population size, initial number immune
    timesteps, population, immune = 300, 1000, 0 

    #min/max degree of vertices
    min_degree, max_degree = 1, 11

    #degree distribution of vertices
    degree_distribution, degree_data = 'random', []
    #degree_distribution, degree_data = 'multimodal', [[2, 0.8], [10, 0.2]] #multimodial distribution where 80% has mean degree 2 and 20% mean degree 10

    #number to divide degree of vertex with when deciding on how many neighbours to meet/timestep
    exp_divide = 3.0

    #probability of edge forming between non neighbouring vertices
    prob_cold_connection = abs(np.random.normal(0.02, 0.008)) 

    #probability of edge forming between vertex of degree 0 and random vertex
    prob_cold_connection_lonely = 0.5 

    #initial number of exposed/infected in population
    initial_num_exposed, initial_num_infected = 0, 1 

    #probability of immigration/emmigration happening at a timestep
    #immigration_prob, emmigration_prob = 0.06, 0.05
    immigration_prob, emmigration_prob = 0, 0

    #immigration and emmigration size by fraction of population
    size_immigration, size_emmigration = 0.07, 0.06 

    #probability that an immigrant is exposed/infectious
    prob_immigrant_exposed, prob_immigrant_infected = 0.0005, 0.0001

    #min/max degree of immigrating vertex
    min_degree_immigrant, max_degree_immigrant = 0, 6 

    #probability that two neighbouring vertices produce offspring at a timestep
    #offspring_prob = 0.001
    offspring_prob = 0

    #scale immigration/emmigration to keep population in vicinity of start value
    scale_immigration_emmigration = True 

    #vaccination strategy, number of doses available
    vaccin_strategy, vaccin_doses = None, 0
    #vaccin_strategy, vaccin_doses = 'random', population * 0.1
    #vaccin_strategy, vaccin_doses = 'acquaintance_vaccination', population * 0.1

    #Number of simulations to run, min fraction of population that must be infected
    num_tests, min_outbreak_size = 20, 0.1








    #lists to store results from simulations
    list_fatalities, sum_R0, list_current_population, list_dist, list_S, list_E, list_I, list_R = [], 0, [], [], [], [], [], []

    #counters for whether or not an sufficiently large outbreak did occur
    outbreak, no_outbreak = 0, 0

    #run indicated number of simulations
    i = 0
    while i < num_tests:
        #run simulation
        vertices, R0, fatalities, current_population, dist, S, E, I, R = runSim(timesteps, population, immune, min_degree, max_degree, prob_cold_connection, prob_cold_connection_lonely, initial_num_exposed, initial_num_infected, immigration_prob, emmigration_prob, size_immigration, size_emmigration, prob_immigrant_exposed, prob_immigrant_infected, min_degree_immigrant, max_degree_immigrant, offspring_prob, scale_immigration_emmigration, vaccin_strategy, vaccin_doses, exp_divide, degree_distribution, degree_data)

        #if the outbreak reaches required size
        if(R[timesteps-1] >= population * min_outbreak_size):
            list_fatalities.append(fatalities)
            sum_R0 += R0
            list_current_population.append(current_population)
            list_dist.append(dist)
            list_S.append(S)
            list_E.append(E)
            list_I.append(I)
            list_R.append(R)
            i += 1
            outbreak += 1
        else:
            no_outbreak += 1

    #variables to store mean of results
    fatalities, current_population, dist, S, E, I, R = {}, {}, {}, {}, {}, {}, {}

    #calculate sum of results at each timesteps
    for i in range(num_tests):
        for j in range(timesteps):
            fatalities[j] = fatalities[j] + list_fatalities[i][j] if j in fatalities else list_fatalities[i][j]
            current_population[j] = current_population[j] + list_current_population[i][j] if j in current_population else list_current_population[i][j]
            S[j] = S[j] + list_S[i][j] if j in S else list_S[i][j]
            E[j] = E[j] + list_E[i][j] if j in E else list_E[i][j]
            I[j] = I[j] + list_I[i][j] if j in I else list_I[i][j]
            R[j] = R[j] + list_R[i][j] if j in R else list_R[i][j]

    for i in range(len(list_dist)):
        for j in range(len([e for e in list_dist[i].values()])):
            dist[j] = dist[j] + list_dist[i][j] if j in dist else list_dist[i][j]

    for i in range(len([e for e in dist.values()])):
        dist[i] /= num_tests

    #calculate mean of results
    R0 = sum_R0 / num_tests
    for i in range(timesteps):
        fatalities[i] /= num_tests
        current_population[i] /= num_tests
        S[i] /= num_tests
        E[i] /= num_tests
        I[i] /= num_tests
        R[i] /= num_tests

    if True:
        plt.xlabel('degree')
        plt.ylabel('Population')
        plt.bar(*zip(*sorted(dist.items())))
        plt.show()

    #7 day moving average of number of deaths
    av_dead = {}
    for day in fatalities:
        start, stop = day - 3, day + 3

        under, over = 0, 0
        if (start < 0):
            under = start
        elif (stop > timesteps):
            over = stop - timesteps

        counter, e_sum = 0, 0
        for i in range(start - under, stop - over):
            e_sum += fatalities[i]
            counter += 1
        av_dead[day] = e_sum / counter

    print(f'estimated R0: {R0}\ncases outbreak: {outbreak}\ncases no outbreak: {no_outbreak}')

    plt.xlabel('Time')
    plt.ylabel('Population')
    #plt.plot(*zip(*sorted(current_population.items())), label='Total population')
    plt.plot(*zip(*sorted(S.items())), 'b', label='Susceptible')
    plt.plot(*zip(*sorted(E.items())), 'y', label='Exposed')
    plt.plot(*zip(*sorted(I.items())), 'r', label='Infectious')
    plt.plot(*zip(*sorted(R.items())), 'g', label='Recovered')
    #plt.gca().set_ylim([0, max(R.values())])
    plt.legend(loc='upper right')
    plt.show()

    plt.xlabel('Time')
    plt.plot(*zip(*sorted(av_dead.items())), label='7 day average deaths')
    plt.legend(loc='upper right')

    
