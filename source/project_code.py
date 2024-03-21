from .building_blocks import *
from cvxopt import matrix, solvers

class SmileAF:
    def __init__(self, strikes, vols, T):
        self.atmvol = vols[int(len(vols)/2)]
        self.fwd = strikes[int(len(strikes)/2)]
        self.T = T
        self.N = 50
        stdev = self.atmvol * math.sqrt(T)
        kmin = self.fwd * math.exp(-0.5*stdev*stdev-5 * stdev)
        kmax = self.fwd * math.exp(-0.5*stdev*stdev+5 * stdev)
        u = (kmax - kmin) / (self.N - 1)
        self.ks = [kmin + u * i for i in range(0, self.N)]
        self.cs = np.zeros(self.N)  # undiscounted call option prices
        self.ps = np.zeros(self.N)  # densities
        self.u = u
        # now we need to construct our constrained optimization problem to solve for cs and ps
        # ... YOUR CODE HERE ... to solve for self.cs and self.ps
        # ...
        
        #x = np.concatenate((self.cs, self.ps))
        # self.cs = [0.1 for i in range(self.N)]
        # self.ps = [1/self.N for i in range(self.N)]

        # def kkt_solver(P, q, G, h, A, b):
        #     n = P.size[0]  # Size of the variables vector
        #     # Implement your custom KKT solver logic here
        #     # For demonstration purposes, we simply return the identity matrix
        #     return matrix(1.0, (n, n))  # Identity matrix as an example

        # # Provide the custom KKT solver to CVXOPT
        # solvers.options['kktsolver'] = kkt_solver


        ## Objective Function

        ## Define R

        print(f'strike: {strikes}')
        print(f'vols: {vols}')
        print(f'u: {u}')
        print(f'ks={self.ks}')
        print(f'T={T}')
        print(f'fwd={self.fwd}')

        R_v = np.zeros((self.N -2 , self.N -2 ))
        R_v[0,0]=2/3
        R_v[0,1]=1/6
        for i in range(1,self.N - 3):
            R_v[i,i-1]=1/6
            R_v[i,i]=2/3
            R_v[i,i+1]=1/6
        R_v[self.N-3,self.N-4]=1/6
        R_v[self.N-3,self.N-3]=2/3
        R_v= self.u * self.u * R_v
        print(R_v.shape)
        print(f'R_v: {R_v}')

        ## Define H
        H_v = np.zeros((2*self.N -2 , 2*self.N -2 ))
        # for i in range(0,self.N): ## temporarily setting diagnol terms to non-zero to make the matrix invertible
        #     H_v[i,i] =  0.000005
        H_v[self.N:,self.N:] = R_v
        print(H_v.shape)
        print(f'HH: {H_v}')

        ### Constraint 1
        
        ## Define Q
        Q_v = np.zeros((self.N -2 , self.N ))
        for i in range(0,self.N - 2):
            Q_v[i,i]     = 1
            Q_v[i,i+1]   = -2
            Q_v[i,i+2]   = 1

        #print(f'Q_vv: {Q_v}')

        ## Define A
        A1_v = np.concatenate((Q_v, -R_v), axis=1)
        b1_v = np.zeros((self.N -2 , 1))

        #print(A1_v)


        ### Constraint 2

        def striketoprice(j):
            k= strikes[j]
            v = vols[j]
            stdev = v * math.sqrt(self.T)
            d1 = (math.log(self.fwd / k)) / stdev + 0.5 * stdev
            d2 = (math.log(self.fwd / k)) / stdev - 0.5 * stdev
            return self.fwd * cnorm(d1) - k * cnorm(d2)
        

        A2_v = np.zeros((len(strikes),2*self.N - 2))
        b2_v = np.zeros((len(strikes), 1))

        for j in range(0,len(strikes)):
            i = bisect.bisect_left(self.ks, strikes[j]) - 1
            k_j = strikes[j]
            k_i_1= self.ks[i+1]
            a=(k_i_1-k_j)/self.u
            b=1-a
            A2_v[j,i] = a
            A2_v[j,i+1] = b
            A2_v[j,self.N -1 + i] = ((a*a*a)-a)*self.u*self.u/6
            A2_v[j,self.N + i] = ((b*b*b)-b)*self.u*self.u/6
            b2_v[j,0] = striketoprice(j)

        ### Constraint 3

        G3_v = np.zeros((self.N -2 , 2*self.N - 2 ))
        for i in range(0,self.N -2):
            G3_v[i,i+self.N] = -1
        
        h3_v = np.zeros((self.N -2,1))


        ### Constraint 4

        A4_v = np.zeros((1,2*self.N - 2))
        for i in range(self.N,2*self.N -2):
            A4_v[0,i]=self.u

        b4_v = np.zeros((1 , 1))
        b4_v[0,0]=1

        ### Constraint 5
        
        A5_v = np.zeros((2,2*self.N - 2))
        A5_v[0,0]=1
        A5_v[1,self.N -1]=1

        b5_v = np.zeros((2 , 1))
        b5_v[0,0] = self.fwd - self.ks[0]
        b5_v[1,0] = 0

        ### Constraint 6

        G6_v = np.zeros((self.N -1 , 2*self.N - 2 ))
        for i in range(0,self.N -1):
            G6_v[i,i] = -1
            G6_v[i,i+1] = 1
        
        h6_v = np.zeros((self.N -1,1))

        ## Optimization

        Q = matrix(2 * H_v)
        p = matrix(np.zeros(2*self.N -2)) ##np.zeros((2*self.N -2 , 1))
        G = matrix(np.concatenate((G3_v, G6_v), axis=0))
        h = matrix(np.concatenate((h3_v, h6_v), axis=0))
        A = matrix(np.concatenate((A1_v, A2_v, A4_v, A5_v), axis=0))
        b = matrix(np.concatenate((b1_v, b2_v, b4_v, b5_v), axis=0))

        #test
        #sol = solvers.qp(Q, p, G, h)
        sol = solvers.qp(Q, p, G, h, A, b)
      

        # H = np.random.rand(self.N, self.N)
        # P = H  # Quadratic term in the objective function
        # q = matrix(0.0, (self.N, 1))  # Linear term in the objective function
        # G = matrix(0.0, (self.N, self.N))  # No inequality constraints
        # h = matrix(0.0, (self.N, 1))  # No inequality constraints

        # # Solve the QP to minimize the objective function
        # sol = solvers.qp(P, q, G, h)

        # Extract the optimal solution
        optimal_solution = sol['x']
        print(f'Optimal_solution: {optimal_solution}')
        
        print('Optimization Done')

        optimal_solution_array = np.array(optimal_solution).T.flatten()
        self.cs = optimal_solution_array[0:self.N]
        self.ps[1:self.N -1] = optimal_solution_array[self.N:]

        # print(f'cs: {self.cs}')
        # print(f'ps: {self.ps}')
        #print(f'ks: {self.ks}')
        #print(f'fwd: {self.fwd}')
        #print(f'Strikes: {strikes}')

        # now we obtained cs and ps, we do not interpolate for price for any k and imply the vol,
        # since at the tails the price to vol gradient is too low and is numerically not stable.
        # Instead, we imply the volatilities for all points between put 10 delta and call 10 delta input points
        # then we make the vol flat at the wings by setting the vols at kmin and kmax,
        # we then construct a cubic spline interpolator on the dense set of volatilities so that it's C2
        # and faster then implying volatilities on the fly.
        # note that this treatment of tail is simplified. It could also introduce arbitrage.
        # In practice, the tails should be calibrated to a certain distribution.
        def implyVol(k, prc, v):
            stdev = v * math.sqrt(self.T)
            d1 = (math.log(self.fwd / k)) / stdev + 0.5 * stdev
            d2 = (math.log(self.fwd / k)) / stdev - 0.5 * stdev
            return self.fwd * cnorm(d1) - k * cnorm(d2) - prc
        khmin = bisect.bisect_left(self.ks, strikes[0])
        print(f'khmin: {khmin}')
        khmax = bisect.bisect_right(self.ks, strikes[len(strikes)-1])
        print(f'khmax: {khmax}')
        kks = [0] * ((khmax+1) - (khmin-1) + 2)
        vs = [0] * ((khmax+1) - (khmin-1) + 2)
        for i in range(khmin-1, khmax+1):
            prc = self.Price(self.ks[i])
            print(f'i: {i}, prc: {prc}')
            f = lambda v: implyVol(self.ks[i], prc, v)
            a, b = 1e-8, 10
            vs[i - (khmin-1) + 1] = optimize.brentq(f, a, b)
            kks[i - (khmin-1) + 1] = self.ks[i]
        kks[0] = kmin
        vs[0] = vs[1]
        kks[len(kks)-1] = kmax
        vs[len(vs)-1] = vs[len(vs)-2]

        #print(f'kks: {kks}')
        #print(f'vs: {vs}')
        self.vs = vs
        self.cubicVol = CubicSpline(kks, vs, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)

    def Vol(self, k):
        if k < self.ks[0]:  # scipy cubicspline bc_type confusing, extrapolate by ourselfs
            return self.vs[0]
        if k > self.ks[-1]:
            return self.vs[-1]
        else:
            return self.cubicVol(k)

    # undiscounted call price - given cs and ps,
    # we can obtain undiscounted call price for any k via cubic spline interpolation
    def Price(self, k):
        if k <= self.ks[0]:
            return self.fwd - k
        if k >= self.ks[self.N-1]:
            return 0.0
        pos = bisect.bisect_left(self.ks, k)
        a = (self.ks[pos] - k) / self.u
        b = 1 - a
        c = (a * a * a - a) * self.u * self.u / 6.0
        d = (b * b * b - b) * self.u * self.u / 6.0
        return a * self.cs[pos-1] + b * self.cs[pos] + c*self.ps[pos-1] + d*self.ps[pos]
        
class SmileCubicSpline:
    def __init__(self, strikes, vols):
        # add additional point on the right to avoid arbitrage
        self.strikes = strikes + [1.1 * strikes[-1] - 0.1 * strikes[-2]]
        self.vols = vols + [vols[-1] + (vols[-1] - vols[-2]) / 10]
        self.cs = CubicSpline(strikes, vols, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)

    def Vol(self, k):
        if k < self.strikes[0]:  # scipy cubicspline bc_type confusing, extrapolate by ourselfs
            return self.vols[0]
        if k > self.strikes[-1]:
            return self.vols[-1]
        else:
            return self.cs(k)

def smileFromMarks(T, S, r, q, atmvol, bf25, rr25, bf10, rr10, smileInterpMethod):
    print(f'T={T}')
    c25 = bf25 + atmvol + rr25/2
    p25 = bf25 + atmvol - rr25/2
    c10 = bf10 + atmvol + rr10/2
    p10 = bf10 + atmvol - rr10/2

    ks = [ strikeFromDelta(S, r, q, T, p10, 0.1, PayoffType.Put),
           strikeFromDelta(S, r, q, T, p25, 0.25, PayoffType.Put),
           S * math.exp((r-q)*T),
           strikeFromDelta(S, r, q, T, c25, 0.25, PayoffType.Call),
           strikeFromDelta(S, r, q, T, c10, 0.1, PayoffType.Call) ]
    # print(T, ks)
    if smileInterpMethod == "CUBICSPLINE":
        return SmileCubicSpline(ks, [p10, p25, atmvol, c25, c10])
    elif smileInterpMethod == "AF":
        return SmileAF(ks, [p10, p25, atmvol, c25, c10], T)