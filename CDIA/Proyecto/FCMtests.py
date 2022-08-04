import numpy as np
import random
import operator
import math

# class FCMTest ():

# 	def __init__():
# 		pass

# 	def MembershipMatrix():
# 		pass

# 	def Parameters():
# 		pass

# 	def Accurracy():
# 		pass

def get_test(test_name, test_params, df):
  if test_name == 'dirichlet': return dirlichet(test_params, df)

class dirlichet():

  def __init__(self, parameters, data_frame):
    self.df = data_frame
    self.n = 150
    self.k = 2
    self.d = 4
    self.m = 2
    self.MAX_ITERS = 12
    self.Parameters(parameters)


  def MembershipMatrix():
    weight = np.random.dirichlet(np.ones(k),n)
    weight_arr = np.array(weight)
    return weight_arr

  def Parameters(self,params_array):
		#setting Params
    self.n = params_array[0]
    self.k = params_array[1]
    self.d = params_array[2]
    self.m = params_array[3]
    self.MAX_ITERS = params_array[4]

		# (len(df), 2, 4, 2, 12)

  def accuracy(self, cluster_labels, class_labels, labels, df):
    correct_pred = 0

    seto = max(set(labels[0:50]), key=labels[0:50].count)
    vers = max(set(labels[50:100]), key=labels[50:100].count)
    virg = max(set(labels[100:]), key=labels[100:].count)

    for i in range(len(df)):
        if cluster_labels[i] == seto and class_labels[i] == 'Iris-setosa':
            correct_pred = correct_pred + 1
        if cluster_labels[i] == vers and class_labels[i] == 'Iris-versicolor' and vers!=seto:
            correct_pred = correct_pred + 1
        if cluster_labels[i] == virg and class_labels[i] == 'Iris-virginica' and virg!=seto and virg!=vers:
            correct_pred = correct_pred + 1

    accuracy = (correct_pred/len(df))*100
    return accuracy


  def initializeMembershipMatrix(self, n, k): # initializing the membership matrix
    membership_mat = []
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]

        flag = temp_list.index(max(temp_list))
        for j in range(0,len(temp_list)):
            if(j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0

        membership_mat.append(temp_list)
    return membership_mat


  def calculateClusterCenter(self, membership_mat, n, k, m, df): # calculating the cluster center
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

  def updateMembershipValue(self, membership_mat, cluster_centers, n, k, m, df): # Updating the membership value
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat

  
