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

        def kkt_solver(P, q, G, h, A, b):
            n = P.size[0]  # Size of the variables vector
            # Implement your custom KKT solver logic here
            # For demonstration purposes, we simply return the identity matrix
            return matrix(1.0, (n, n))  # Identity matrix as an example

        # Provide the custom KKT solver to CVXOPT
        solvers.options['kktsolver'] = kkt_solver

        H = np.zeros((2*self.N -2 , 2*self.N -2 ))
        H[self.N,self.N]=2/3
        H[self.N,self.N+1]=1/6
        for i in range(self.N + 1,2*self.N - 3):
            H[i,i-1]=1/6
            H[i,i]=2/3
            H[i,i+1]=1/6
        H[2*self.N-3,2*self.N-4]=1/6
        H[2*self.N-3,2*self.N-3]=2/3
        
        #print(H)
        Q = 2 * H
        p = np.zeros((2*self.N -2 , 1))
        G = matrix([[-1.0, 0.0], [0.0, -1.0]])
        h = matrix([0.0, 0.0])
        A = matrix([1.0, 1.0], (1, 2))
        b = matrix(1.0)
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