from .building_blocks import *
from cvxopt import matrix, solvers

class SmileAF:
    def __init__(self, strikes, vols, T,):
        self.atmvol = vols[int(len(vols)/2)]
        self.fwd = strikes[int(len(strikes)/2)]
        self.T = T
        self.N = 50
        stdev = self.atmvol * math.sqrt(T)
        kmin = self.fwd * math.exp(-0.5*stdev*stdev-5 * stdev)
        kmax = self.fwd * math.exp(-0.5*stdev*stdev+5 * stdev)
        u = (kmax - kmin) / (self.N - 1)
        
        # calculating the range of ks
        self.ks = [kmin + u * i for i in range(0, self.N)]
        self.cs = np.zeros(self.N)  # undiscounted call option prices
        self.ps = np.zeros(self.N)  # densities
        self.u = u
        # now we need to construct our constrained optimization problem to solve for cs and ps
        # ... YOUR CODE HERE ... to solve for self.cs and self.ps
        # ...
    
        # Constraint 1
        q = np.zeros((self.N-2, self.N))
        r = np.zeros((self.N-2, self.N-2))
        for i in range(self.N-2):
            # setting cublic spline constraint for Q
            q[i][i] = 1
            q[i][i+1] = -2
            q[i][i+2] = 1

            # setting cubic spline constaint for R
            if i == 0:
                r[i][i] = 2/3
                r[i][i+1] = 1/6
            elif i == self.N-3:
                r[i][-1] = 2/3
                r[i][-2] = 1/6
            else:
                r[i][i-1] = 1/6
                r[i][i] = 2/3
                r[i][i+1] = 1/6

        r = r * u**2
        negativeR = r * -1
        c1 = np.concatenate((q, negativeR), axis=1)
        c1v =np.zeros((self.N-2,1))
        
        # Constraint 2
        c2 = np.zeros((5,2*self.N-2))
        c2v =np.zeros((5,1))
        
        for i in range(5):
            payoffType = PayoffType.Call
            k = strikes[i]
            vol = vols[i]
            secTempPosC = bisect.bisect_left(self.ks, k)
            firstTempPosC = secTempPosC - 1
            firstTempPosP = firstTempPosC + 49
            secTempPosP = secTempPosC + 49

            # calculation of C(kj) using black scholes
            # tempPrice = bsPrice (self.S, self.r, self.q, vol, T, k, payoffType)
            tempStdDev = vol * np.sqrt(T)
            d1 = math.log(self.fwd / k) / tempStdDev + tempStdDev / 2
            d2 = d1 - tempStdDev
            tempPrice = (self.fwd * cnorm(d1) - cnorm(d2) * k)
            a = (self.ks[secTempPosC] - k) / self.u
            b = 1 -a 
            c = (a * a * a - a) * self.u * self.u / 6.0
            d = (b * b * b - b) * self.u * self.u / 6.0

            # setting invididual constraint
            c2[i][firstTempPosC] = a
            c2[i][secTempPosC] = b
            c2[i][firstTempPosP] = c
            c2[i][secTempPosP] = d

            c2v[i] = tempPrice
        
        # Constraint 3
        c3 = np.zeros((self.N-2, 2*self.N-2))
        tempCounter = 50
        for i in range (self.N-2):
            c3[i][tempCounter] = -1
            tempCounter += 1
        c3v = np.zeros((self.N-2,1))
        
        # Constraint 4
        c4 = np.zeros((1, 2*self.N-2))
        for i in np.arange(self.N, 2*self.N-2):
            c4[0][i] = self.u
        c4v = np.array([[1]])
      
        # Constraint 5
        c5 = np.zeros((2, 2*self.N-2))
        c5[0][0] = 1
        c5[1][self.N-1] = 1
        c5v = np.zeros((2,1))
        c5v[0][0] = self.fwd-kmin
        
        # Constraint 6
        c6 = np.zeros((self.N-1, 2*self.N-2))
        c6v = np.zeros((self.N-1,1))
        for i in range(self.N-1):
            c6[i][i] = -1
            c6[i][i+1] = 1
        
        # objective function
        H = np.zeros((2*self.N-2, 2*self.N-2))
        tempI = 0
        
        H[self.N:, self.N:] = r
        
        # changing to cvxopt format
        P = matrix(2*H, tc='d')
        q = matrix(np.zeros((2*self.N-2)), tc='d')
        #G = matrix(c3, tc='d')
        #h = matrix(c3v, tc='d')
        G = matrix(np.concatenate((c3, c6), axis=0), tc='d')
        h = matrix(np.concatenate((c3v, c6v), axis=0), tc='d')

        args1 = (c1, c2, c4,c5)
        args2 = (c1v, c2v, c4v,c5v)
        
        A = matrix( np.concatenate(args1), tc='d')
        b = matrix( np.concatenate(args2), tc='d')

        sol = solvers.qp(P,q,G,h,A,b)
        self.cs = sol['x'][0:self.N] # undiscounted call option prices
        self.ps = self.ps.reshape(50,1)
        self.ps[1:-1] = sol['x'][self.N:2*self.N-2] # densities
        #sol = solvers.qp(P,q,G,h)
        
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
        khmax = bisect.bisect_right(self.ks, strikes[len(strikes)-1])
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