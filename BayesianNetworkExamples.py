# Mark Trinquero
# Probabilistic modeling using PythonBayesianNetworkToolbox
# Bayesian Networks and Gibbs Sampling

# ---------------------------------------------------------------------------------------------------
# Resources Consulted:
# [PythonBayesianNetworkToolbox](https://github.com/thejinxters/PythonBayesianNetworkToolbox)
# http://www.cs.cmu.edu/afs/cs/academic/class/15381-s07/www/slides/032907bayesNets2.pdf
# http://sites.nicholas.duke.edu/statsreview/probability/jmc/
# https://en.wikipedia.org/wiki/Bayes%27_theorem
# ---------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------
# Bayesian Network - Power Plant System Example
# ---------------------------------------------------------------------------------------------------

from Node import BayesNode
from Graph import BayesNet

def make_power_plant_net():
    # intial setup
    nodes = []
    numberOfNodes = 5
    temperature = 0
    gauge = 1
    faulty_gauge = 2
    alarm = 3
    faulty_alarm = 4

    # create the nodes
    T_node      = BayesNode(0, 2, name='temperature')
    G_node      = BayesNode(1, 2, name='gauge')
    F_G_node    = BayesNode(2, 2, name='faulty gauge')
    A_node      = BayesNode(3, 2, name='alarm')
    F_A_node    = BayesNode(4, 2, name='faulty alarm')

    # temperature
    T_node.add_child(G_node)
    T_node.add_child(F_G_node) 

    # faulty gauge
    F_G_node.add_child(G_node)
    F_G_node.add_parent(T_node)

    # gauge
    G_node.add_parent(T_node)
    G_node.add_parent(F_G_node)
    G_node.add_child(A_node)

    # faulty alarm
    F_A_node.add_child(A_node)

    # alarm 
    A_node.add_parent(G_node)
    A_node.add_parent(F_A_node)

    # add the nodes for setting up example network
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]
    return BayesNet(nodes)



# ---------------------------------------------------------------------------------------------------
# Example Setup with given probabilities
# ---------------------------------------------------------------------------------------------------

# 1. The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty. For simplicity, say that the gauge's "true" value corresponds with its "hot" reading and "false" with its "normal" reading, so the gauge would have a 95% chance of returning "true" when the temperature is hot and it is not faulty.
# 2. The alarm is faulty 15% of the time.
# 3. The temperature is hot (call this "true") 20% of the time.
# 4. When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
# 5. The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds correctly to the gauge 90% of the time when the alarm is not faulty. For instance, when it is faulty, the alarm sounds 55% of the time that the gauge is "hot" and remains silent 55% of the time that the gauge is "normal."

# Note: 0 represents the index of the false probability, and 1 represents true.

from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution

def set_probability(bayes_net):    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    # Use the following Boolean variables in your implementation:
    #       A       = alarm sounds
    #       F_A     = alarm is faulty
    #       G       = gauge reading (high = True, normal = False)
    #       F_G     = gauge is faulty
    #       T       = actual temperature (high = True, normal = False)

    # temperature node distribution
    # 0 Index = False Prob
    # 1 Index = True Prob
    # True = Hot Temp = 20%
    # False = Normal Temp = 80% 
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([],[])
    T_distribution[index] = [0.8,0.2]
    T_node.set_dist(T_distribution)


    # faulty alarm node distribution
    # 0 Index = False Prob
    # 1 Index = True Prob
    # True = alarm is faulty = 15%
    # False = alarm is not faulty = 85% 
    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([],[])
    F_A_distribution[index] = [0.85,0.15]
    F_A_node.set_dist(F_A_distribution)


    # faulty gauge node distribution
    # 0 column -- when temp is normal (T = False)
    # 1 Index -- when temp is hot (T = True)
    # True = faulty alarm, False = not faulty alarm
    # when Temp is normal (T=F), 
    #      F_G = false = .95    normal alarm with normal temp
    #      F_G = true = .05    faulty alarm with normal temp
    # when Temp is hot (T=T), 
    #      F_G = false = .20    normal alarm with hot temp
    #      F_G = true = .80    faulty alarm with hot temp
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)
    dist[0,:] = [0.95, 0.05]
    dist[1,:] = [0.20, 0.80]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)


    # gauge node distribution
    #Temp:  hot= T     normal = F
    #F_G:   faulty= T  not faulty/normal = F
    #G:     hot = T    normal = F
    # Temp   F_G   P(G=true|Temp,F_G)
    # T      T     0.20
    # T      F     0.95
    # F      T     0.80
    # F      F     0.05
    # Temp = Hot = True
    # F_G = Faulty = True
    # True, True, True = .20
    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[1,1,:] = [0.80, 0.20]
    dist[1,0,:] = [0.05, 0.95]
    dist[0,1,:] = [0.20, 0.80]
    dist[0,0,:] = [0.95, 0.05]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)


    # alarm node distribution 
    #Gauge:     hot= T          normal = F
    #F_A:       faulty= T       not faulty/normal = F
    #A:         sounds = T      alarm doesnt sound = F
    # Gauge  F_A   P(A=true|G,F_A)
    # T      T     0.55
    # T      F     0.90
    # F      T     0.45
    # F      F     0.10
    # Gauge = Hot = True
    # F_A = Faulty = True
    # True, True, True = .55
    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    dist[1,1,:] = [0.45, 0.55]
    dist[1,0,:] = [0.10, 0.90]
    dist[0,1,:] = [0.55, 0.45]
    dist[0,0,:] = [0.90, 0.10]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    return bayes_net




# ---------------------------------------------------------------------------------------------------
# Probabiltiy Calculations and Testing
# ---------------------------------------------------------------------------------------------------
from Inference import JunctionTreeEngine


def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal probability of the alarm ringing (T/F) in the power plant system."""
    # True or False
    alarm_rings = alarm_rings
    # set up engine
    engine = JunctionTreeEngine(bayes_net)
    # get nodes
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    G_node = bayes_net.get_node_by_name('gauge')
    A_node = bayes_net.get_node_by_name('alarm')
    # Compute the marginal probability of alarm given no evidence
    Q = engine.marginal(A_node)[0]
    #Q is a DiscreteDistribution, and so to index into it, we have to use the class' method to create an index
    index = Q.generate_index([False], range(Q.nDims))
    alarm_prob_false = Q[index]
    alarm_prob_true = float(1- Q[index])
    alarm_prob = alarm_prob_true
    print "The marginal probability of alarm=true:", alarm_prob_true
    # marginal probablility of alarm ringing
    return alarm_prob



def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal probability of the gauge showing hot (T/F) in the power plant system."""
    # True or False
    gauge_hot = gauge_hot
    # Set up engine
    engine = JunctionTreeEngine(bayes_net)
    # Get nodes
    G_node = bayes_net.get_node_by_name('gauge')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    T_node = bayes_net.get_node_by_name('temperature')
    # Compute the marginal probability of gauge given no evidence
    Q = engine.marginal(G_node)[0]
    #Q is a DiscreteDistribution, and so to index into it, we have to use the class' method to create an index
    index = Q.generate_index([False], range(Q.nDims))
    gauge_prob_false = Q[index]
    gauge_prob_true = float(1- Q[index])
    gauge_prob = gauge_prob_true
    print "The marginal probability of gauge=true:", gauge_prob
    # marginal probablility of gauge showing hot
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate theprobability of the temperature being hot (T/F) in the power plant system, given that the alarm sounds and neither the gauge nor alarm is faulty."""
    engine = JunctionTreeEngine(bayes_net)
    G_node = bayes_net.get_node_by_name('gauge')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    T_node = bayes_net.get_node_by_name('temperature')
    A_node = bayes_net.get_node_by_name('alarm')    
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    # True of False
    temp_hot = temp_hot
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    engine.evidence[A_node] = True
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([False],range(Q.nDims))
    temp_prob_false = Q[index]
    temp_prob_true = float(1- Q[index])
    temp_prob = temp_prob_true
    print "The probability of temperature=true (hot) given,"
    print "F_A = false, F_G = false, and alarm sounds"
    print "temp_prob:  ",  temp_prob
    # the probability that the temperature is actually hot, given that the alarm sounds and the alarm and gauge are both working
    return temp_prob




# ---------------------------------------------------------------------------------------------------
# Gibbs Sampling setup and example testing
# ---------------------------------------------------------------------------------------------------
# 3 teams : A, B and C. With 3 total matches are played. 

#Assume the following variable conventions: 
# | variable name          	| description|
# |A 						| A's skill level|
# |B 						| B's skill level|
# |C 						| C's skill level|
# |AvB 						| the outcome of A vs. B <br> (0 = A wins, 1 = B wins, 2 = tie)|
# |BvC 						| the outcome of B vs. C <br> (0 = B wins, 1 = C wins, 2 = tie)|
# |CvA 						| the outcome of C vs. A <br> (0 = C wins, 1 = A wins, 2 = tie)|
# 
# Assume that each team has the following prior distribution of skill levels: 
# |skill levels 	|P(skill level)|
# |0 				|0.15|
# |1 				|0.45|
# |2 				|0.30|
# |3 				|0.10|

# Assume that the differences in skill levels correspond to the following probabilities of winning:
# | skill difference (T2 - T1) 		| T1 wins 		| T2 wins		| Tie |
# |0 								|0.10 			|0.10 			|0.80|
# |1 								|0.20 			|0.60 			|0.20|
# |2 								|0.15 			|0.75 			|0.10|
# |3 								|0.05 			|0.90 			|0.05|


def get_game_network():
    # intial setup for problem described above
    nodes = []
    numberOfNodes = 6
    A = 0
    B = 1
    C = 2
    AvB = 3
    BvC = 4
    CvA = 5

    # create the nodes
    A_node      = BayesNode(0, 4, name='A')
    B_node      = BayesNode(1, 4, name='B')
    C_node      = BayesNode(2, 4, name='C')
    AvB_node    = BayesNode(3, 3, name='AvB')
    BvC_node    = BayesNode(4, 3, name='BvC')
    CvA_node    = BayesNode(5, 3, name='CvA')

    # setup A Node prior distribution
    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([],[])
    A_distribution[index] = [0.15,0.45, 0.3, 0.1]
    A_node.set_dist(A_distribution)

    # setup B Node prior distribution
    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = [0.15,0.45, 0.3, 0.1]
    B_node.set_dist(B_distribution)

    # setup C Node prior distribution
    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([],[])
    C_distribution[index] = [0.15,0.45, 0.3, 0.1]
    C_node.set_dist(C_distribution)

    # Probabilty Table for Matchup of Two Teams based on Skill Difference 
    # | skill difference (T2 - T1)      | T1 wins       | T2 wins       | Tie |
    # |0                                |0.10           |0.10           |0.80|
    # |1                                |0.20           |0.60           |0.20|
    # |2                                |0.15           |0.75           |0.10|
    # |3                                |0.05           |0.90           |0.05|
    # |-1                               |0.60           |0.20           |0.20|
    # |-2                               |0.75           |0.15           |0.10|
    # |-3                               |0.90           |0.05           |0.05|

    # setup AvB Node distribution
    AvB_distribution = DiscreteDistribution(AvB_node)
    dist = zeros([A_node.size(), B_node.size(), AvB_node.size()], dtype=float32)
    for a in range(A_node.size()):
        for b in range(B_node.size()):
            if (b - a) == -3:
                dist[a,b,:] = [0.90, 0.05, 0.05]
            elif (b - a) == -2:
                dist[a,b,:] = [0.75, 0.15, 0.10]
            elif (b - a) == -1:
                dist[a,b,:] = [0.60, 0.20, 0.20]
            elif (b - a) == 0:
                dist[a,b,:] = [0.10, 0.10, 0.80]
            elif (b - a) == 1:
                dist[a,b,:] = [0.20, 0.60, 0.20]
            elif (b - a) == 2:
                dist[a,b,:] = [0.15, 0.75, 0.10]
            elif (b - a) == 3:
                dist[a,b,:] = [0.05, 0.90, 0.05]
            else:
                print "ERROR in AvB node setup"
    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AvB_node], table=dist)
    AvB_node.set_dist(AvB_distribution)


    # setup BvC Node distribution
    BvC_distribution = DiscreteDistribution(BvC_node)
    dist = zeros([B_node.size(), C_node.size(), BvC_node.size()], dtype=float32)
    for b in range(B_node.size()):
        for c in range(C_node.size()):
            if (c - b) == -3:
                dist[b,c,:] = [0.90, 0.05, 0.05]
            elif (c - b) == -2:
                dist[b,c,:] = [0.75, 0.15, 0.10]
            elif (c - b) == -1:
                dist[b,c,:] = [0.60, 0.20, 0.20]
            elif (c - b) == 0:
                dist[b,c,:] = [0.10, 0.10, 0.80]
            elif (c - b) == 1:
                dist[b,c,:] = [0.20, 0.60, 0.20]
            elif (c - b) == 2:
                dist[b,c,:] = [0.15, 0.75, 0.10]
            elif (c - b) == 3:
                dist[b,c,:] = [0.05, 0.90, 0.05]
            else:
                print "ERROR in BvC node setup"
    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BvC_node], table=dist)
    BvC_node.set_dist(BvC_distribution)


    # setup CvA Node distribution
    CvA_distribution = DiscreteDistribution(CvA_node)
    dist = zeros([C_node.size(), A_node.size(), CvA_node.size()], dtype=float32)
    for c in range(C_node.size()):
        for a in range(A_node.size()):
            if (a - c) == -3:
                dist[c,a,:] = [0.90, 0.05, 0.05]
            elif (a - c) == -2:
                dist[c,a,:] = [0.75, 0.15, 0.10]
            elif (a - c) == -1:
                dist[c,a,:] = [0.60, 0.20, 0.20]
            elif (a - c) == 0:
                dist[c,a,:] = [0.10, 0.10, 0.80]
            elif (a - c) == 1:
                dist[c,a,:] = [0.20, 0.60, 0.20]
            elif (a - c) == 2:
                dist[c,a,:] = [0.15, 0.75, 0.10]
            elif (a - c) == 3:
                dist[c,a,:] = [0.05, 0.90, 0.05]
            else:
                print "ERROR in CvA node setup"
    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A_node, CvA_node], table=dist)
    CvA_node.set_dist(CvA_distribution)


    # Setup Network (Parents & Children)
    # A
    A_node.add_child(AvB_node)
    A_node.add_child(CvA_node)

    # B
    B_node.add_child(AvB_node)
    B_node.add_child(BvC_node)

    # C
    C_node.add_child(BvC_node)
    C_node.add_child(CvA_node)

    # AvB
    AvB_node.add_parent(A_node)
    AvB_node.add_parent(B_node)

    # BvC 
    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)

    # CvA 
    CvA_node.add_parent(C_node)
    CvA_node.add_parent(A_node)

    # add the nodes for setting up network
    nodes = [A_node, B_node, C_node, AvB_node, BvC_node, CvA_node]
    return BayesNet(nodes)





# ---------------------------------------------------------------------------------------------------
# Calculate posterior distribution for the 3rd match
# ---------------------------------------------------------------------------------------------------
from Inference import EnumerationEngine

# Suppose that you know the following outcome of two of the three games: A beats B and A draws with C. 
# Calculate the posterior distribution for the outcome of the BvC match in calculate_posterior(). Use EnumerationEngine ONLY. 

def calculate_posterior(games_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""

    # list of probabilities 
    # Win, Loss, Tie likelihoods
    posterior = [0,0,0]
    engine = EnumerationEngine(games_net)
    A_node = games_net.get_node_by_name('A')
    B_node = games_net.get_node_by_name('B')
    B_node = games_net.get_node_by_name('B')
    AvB_node = games_net.get_node_by_name('AvB')    
    BvC_node = games_net.get_node_by_name('BvC')
    CvA_node = games_net.get_node_by_name('CvA')

    # |AvB      the outcome of A vs. B <br> (0 = A wins, 1 = B wins, 2 = tie)|
    # |BvC      the outcome of B vs. C <br> (0 = B wins, 1 = C wins, 2 = tie)|
    # |CvA      the outcome of C vs. A <br> (0 = C wins, 1 = A wins, 2 = tie)|
    #GIVEN: A beats B
    #GIVEN: A draws with C
    #engine = EnumerationEngine(games_net)
    # A beats B
    engine.evidence[AvB_node] = [1.0, 0.0, 0.0]
    # A ties with C
    engine.evidence[CvA_node] = [0.0, 0.0, 1.0]
    # no evidence
    engine.evidence[BvC_node] = [0.0, 0.0, 0.0]
    print engine
    print BvC_node.dist.table
    #Q = engine.marginal(BvC_node)[0,0,0]
    Q = engine.marginal(BvC_node)[0]
    #AvB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AvB_node], table=dist)
    print '-'*10
    print Q
    print enumerate(Q)
    print '-'*10
    posterior = Q
    return posterior



# ---------------------------------------------------------------------------------------------------
# GIBBS Sampling
# ---------------------------------------------------------------------------------------------------
def Gibbs_sampling(games_net, initial_value, number_of_teams=5):
    A= games_net.get_node_by_name("A")      
    AvB= games_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_value)
    random_val = random.randint(0,10)
    val = initial_value[random_val]
    return sample


## TESTING
number_of_teams=5
n = number_of_teams
initial_state = [0]*(2*n)
sample = Gibbs_sampling(game_net, initial_state, number_of_teams=5)




# ---------------------------------------------------------------------------------------------------
# GIBBS Sampling for convergence
# ---------------------------------------------------------------------------------------------------
def converge_count_Gibbs(bayes_net, initial_state, match_results, number_of_teams=5):
    count=0
    prob_win = 0.0
    prob_loss = 0.0
    prob_tie = 0.0
    posterior = [prob_win,prob_loss,prob_tie]
    convergence_level = 0.001
    prev = []
    curr = []
    return count,posterior


## TESTING
from random import randint,uniform
#initial_state:
match_results = [0,0,1,1]
converge_count_Gibbs(game_net, initial_state, match_results, number_of_teams=5)
