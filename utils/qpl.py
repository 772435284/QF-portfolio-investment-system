from typing import List
import numpy as np
import math

def get_nqpr(DT_CL: List[float]) -> List[float]:
	# print(DT_CL)
	K = []
	for eL in range(0,21):
		K.append(pow((1.1924 + (33.2383*eL) + (56.2169*eL*eL))/(1 + (43.6106 *eL)),1/3))

	DT_RT = []
	for d in range(len(DT_CL) - 1):
		DT_RT.append(DT_CL[d]/DT_CL[d+1] if DT_CL[d+1]>0 else 1)

	# 计算平均值和标准差
	mu = np.mean(DT_RT)
	sigma = np.std(DT_RT)

	# Approximately dr
	dr = (3 * sigma) / 50

	auxR = 0
	Q = [0 for i in range(100)]
	NQ = [0 for i in range(100)]
	maxRno = len(DT_RT)
	tQno = 0
	nQ = 0
	numOfRInQSlot = 0

	for r in DT_RT:
		bFound = False
		nQ = 0
		auxR = 1 - (dr * 50) # Left boundary
		while ((bFound != True) and (nQ < 100)):
			if r > auxR and r <= (auxR + dr):
				Q[nQ]+=1
				tQno+=1
				bFound = True
			else:
				nQ+=1
				auxR = auxR + dr

	# Normalize
	NQ = np.array(Q)/tQno

	# Find MaxQ and index of MaxQ
	maxQ = NQ.max()

	maxQno = NQ.tolist().index(maxQ)
	r=[]
	for i in range(100):
		r.append(1-50*dr+i*dr)

	r0  = r[maxQno] - (dr/2)
	r1  = r0 + dr
	rn1 = r0 - dr
	Lup = (pow(rn1,2)*NQ[maxQno-1])-(pow(r1,2)*NQ[maxQno+1])
	Ldw = (pow(rn1,4)*NQ[maxQno-1])-(pow(r1,4)*NQ[maxQno+1])
	L   = abs(Lup/Ldw)

	QFEL = [0. for i in range(21)]
	QPR = [0. for i in range(21)]
	NQPR = [0. for i in range(21)]

	# Cardano Method
	for eL in range(0, 21):
		p = -1 * pow((2*eL+1),2);
		q = -1 * L * pow((2*eL+1),3) * pow(K[eL],3)

		# Apply Cardano's Method to find the real root of the depressed cubic equation
		u = pow((-0.5*q + math.sqrt(((q*q/4.0) + (p*p*p/27.0)))),1/3)
		v = pow((-0.5*q - math.sqrt(((q*q/4.0) + (p*p*p/27.0)))),1/3)

		QFEL[eL] = u+v

	for eL in range(0, 21):
		QPR[eL]  = QFEL[eL]/QFEL[0]
		NQPR[eL] = 1 + 0.21*sigma*QPR[eL]

	return NQPR
