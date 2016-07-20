Continuous, KDE based Lassiter and Goodman 2013. Super slow and inaccurate, ***do not use***.

~~~~
var COST = 2/3
var RATIONALITY = 4

var kdeScore = cache(function(dist, value, sigma) {  
  return Math.log(expectation(Infer({ method: 'MCMC', samples:1000}, function() {
    var x = sample(dist);
    return Math.exp(Gaussian({ mu: x, sigma: sigma }).score(value));
  })));
});

var worldPrior = function() {
    return sample(Gaussian({mu:0.5, sigma:0.3}));
};

var thresholdPrior = function() {
    return sample(Uniform({a:0, b:1}));
};

var utterancePrior = function() {
    var lexicon = ["", "tall", "short"];
    return lexicon[randomInteger(lexicon.length)];
};

var semantics = function(u, w, t) {
    return u == "tall" ? w > t : 
           u == "short"? w < t :
           true;
};
var cost = function(u) {
    return u == ""? 0 : COST;
};

var LiteralListener = cache(function(utterance, threshold) {
    Infer({method: 'rejection', samples:1000},
          function(){
              var world = worldPrior();
              condition(semantics(utterance, world, threshold));
              return world;
          }
    );
});

var L1Speaker = cache(function(world, threshold) {
    Infer({method: 'enumerate'},
          function() {
              var utterance = utterancePrior();
              var utility = kdeScore(LiteralListener(utterance, threshold), world, .05);
              factor(RATIONALITY * (utility - cost(utterance)));
              return utterance;
          }
    );
});

var L1Listener = function(utterance) {
    Infer({method: 'MCMC', samples:1000, burn:200},
          function() {
              var world = worldPrior();
              var threshold = thresholdPrior();
              factor(L1Speaker(world, threshold).score(utterance));
              return {world:world, threshold:threshold};
          }
    );
};
viz.marginals(L1Listener("tall"));
~~~~

Lassiter and Goodman 2013 with discrete distributions

~~~~
var COST = 2
var RATIONALITY = 4
var prec = 100

// Generate array of size N+1 of 0...N
var genArray = function(N) {
  return N > 0? genArray(N-1).concat(N) : [0];
}

// Discretized Distributions
var DiscreteUniform = function(N) {
  return Discrete({ps:map(function() {1/(N+1)}, genArray(N))})
};

var DiscreteNormal = function(N) {
  return Discrete({ps:map(function(k) {Math.exp(Gaussian({mu:0.25, sigma:0.15}).score(k/(N)))}, 
                          genArray(N))});
}

var DiscreteBeta = function(N, a, b) {
  // This ones a bit weird - ignores endpoints w/ prob 0
  return Discrete({ps:map(function(k) {Math.exp(Beta({a:a, b:b}).score((k+1)/(N+2)))}, 
                          genArray(N))});
}

// Priors
var worldPrior = function() {
    return (sample(DiscreteBeta(prec, 1, 9)))/prec;
};

var thresholdPrior = function() {
    return (sample(DiscreteUniform(prec)))/prec;
};

var utterancePrior = function() {
    var lexicon = ["", "safe", "dangerous"];
    return lexicon[randomInteger(lexicon.length)];
};


// Literal meaning and cost
var semantics = function(u, w, t) {
    return u == "dangerous" ? w >= t : 
           u == "safe" ? w <= t :
           true;
};
var cost = function(u) {
    return u == ""?  0 : COST;
};

// Agents
var LiteralListener = cache(function(utterance, threshold) {
    Infer({method: 'enumerate'},
          function(){
              var world = worldPrior();
              condition(semantics(utterance, world, threshold));
              return world;
          }
    );
});

var L1Speaker = cache(function(world, threshold) {
    Infer({method: 'enumerate'},
          function() {
              var utterance = utterancePrior();
              var utility = LiteralListener(utterance, threshold).score(world)
              factor(RATIONALITY * (utility - cost(utterance)));
              return utterance;
          }
    );
});

var L1Listener = function(utterance) {
    Infer({method: 'enumerate'},
          function() {
              var world = worldPrior();
              var threshold = thresholdPrior();
              factor(L1Speaker(world, threshold).score(utterance));
              return {world:world, threshold:threshold};
          }
    );
};

//Visualization
viz.auto(Infer({method:'enumerate'}, worldPrior));
viz.marginals(L1Listener("safe"));
~~~~

Granularity as the variance of a normal Distribution used to compute "KDE score" of literal listener.



~~~~
var COST = 2/3
var RATIONALITY = 4
var prec = 100

// Generate array of size N+1 of 0...N
var genArray = function(N) {
  return N > 0? genArray(N-1).concat(N) : [0];
}

// Discretized Distributions
var DiscreteUniform = cache(function(N) {
  return Discrete({ps:map(function() {1/(N+1)}, genArray(N))})
});

var DiscreteNormal = cache(function(N, m, s) {
  return Discrete({ps:map(function(k) {Math.exp(Gaussian({mu:m, sigma:s}).score(k/(N)))}, 
                          genArray(N))});
});

var DiscreteBeta = cache(function(N, a, b) {
  // This ones a bit weird - ignore endpoints w/ prob 0
  return Discrete({ps:map(function(k) {Math.exp(Beta({a:a, b:b}).score((k+1)/(N+2)))}, 
                          genArray(N))});
});

// Priors
var worldPrior = function() {
    return (sample(DiscreteBeta(prec, 1, 9)))/prec;
};

var utterancePrior = function() {
    var lexicon = ["", "safe", "dangerous"];
    return lexicon[randomInteger(lexicon.length)];
};

var granularityPrior = function() {
    return (sample(DiscreteNormal(prec, 0.2, 0.15)) + 1)/prec;
}


// Literal meaning and cost - threshold fixed at max
var semantics = function(u, w) {
    return u == "dangerous" ? w >0 : 
           u == "safe"? w == 0 :
           true;
};

var cost = function(u) {
    return u == ""? 0 : COST;
};

// Agents
var LiteralListener = cache(function(utterance) {
    Infer({method: 'enumerate'},
          function(){
              var world = worldPrior();
              condition(semantics(utterance, world));
              return world;
          }
    );
});

var L1Speaker = cache(function(world, granularity) {
    Infer({method: 'enumerate'},
          function() {
              var utterance = utterancePrior();
              // Generate a Normal centered at a sample from the literal listener (IE, what the listener would interpret)
              // with variance equal to the granularity (how far the listener can be from the truth before we mind) and use the log-prob
              // to weight the choice of utterance.
              var utility = Gaussian({mu:sample(LiteralListener(utterance)), sigma:granularity}).score(world);
              factor(RATIONALITY * (utility - cost(utterance)));
              return utterance;
          }
    );
});

var L1Listener = function(utterance) {
    Infer({method: 'enumerate'},
          function() {
              var world = worldPrior();
              var granularity = granularityPrior();
              factor(L1Speaker(world, granularity).score(utterance));
              return {world:world, granularity:granularity};
          }
    );
};

//Visualization
viz.auto(Infer({method:'enumerate'}, worldPrior));
viz.marginals(L1Listener("safe"));
~~~~

Kao et al faithfully.

~~~~
var COST = 2
var RATIONALITY = 4
var prec = 10

// Generate array of size N+1 of 0...N
var genArray = function(N) {
  return N > 0? genArray(N-1).concat(N) : [0];
}

// Generate all possible world/affect pairs
var cross = function(x) {return cross_(x,x)}
var cross_ = function(x, y) {
    return x >= 0? zip(repeat(y + 1, function() {return(x)}), genArray(y)).concat(cross_(x - 1, y)) :
                    [];
                  
}

var objectify = function(x, prec1, prec2) {
  return {world:x[0]/prec1, affect:x[1]/prec2}
}

var states = map(function(x) {objectify(x, prec, 1)}, cross_(prec, 1));

// Why do I feel like I'm rewriting the standard library of any reasonable language???
var equal = function(a,b) {
  if (a.length != b.length) 
    return false;
  if (a.length == 0)
    return true
  return a[0] == b[0]? equal(a.slice(1), b.slice(1)) : false;
}

// Discretized Distributions
var DiscreteUniform = function(N) {
  return Discrete({ps:map(function() {1/(N+1)}, genArray(N))})
};

var DiscreteNormal = function(N, m, s) {
  return Discrete({ps:map(function(k) {Math.exp(Gaussian({mu:m, sigma:s}).score(k/(N)))}, 
                          genArray(N))});
};

var DiscreteBeta = function(N, a, b) {
  // This ones a bit weird - ignore endpoints w/ prob 0
  return Discrete({ps:map(function(k) {Math.exp(Beta({a:a, b:b}).score((k+1)/(N+2)))}, 
                          genArray(N))});
};

// Priors
var worldPrior = function() {
    return (sample(DiscreteNormal(prec, 0.25, 1)))/prec;
};

var lexicon = ["", "some", "all"];
var utterancePrior = function() {
    return lexicon[randomInteger(lexicon.length)];
};

var affectPrior = function(world) {
    return flip((0.85 * world) + 0.10)? 1 : 0;
}
var goals = [function(w,a) {return [w]},
             function(w,a) {return [a]},
             function(w,a) {return [w, a]}];

var goalPrior = function() {
    return goals[randomInteger(goals.length)];
};


// Literal meaning and cost - threshold fixed at max
var semantics = function(u, w) {
    return u == "all" ? w == 1 : 
           u == "some"? w > 0 :
           true;
};

var cost = function(u) {
    return u == ""? 0 : COST;
}

// Agents

var LiteralListener = cache(function(utterance) {
    Infer({method: 'enumerate'},
          function(){
              var world = worldPrior();
              var affect = affectPrior(world);
              condition(semantics(utterance, world));
              return {world:world, affect:affect};
          }
    );
});

var L1Speaker = (function(world, affect, goal) {
    Infer({method: 'enumerate'},
          function() {
              var utterance = utterancePrior();
              var l0_interp = sample(LiteralListener(utterance));
              condition(equal(goal(l0_interp.world, l0_interp.affect), goal(world, affect)))
              return utterance;
          }
    );
});

var L1Listener = function(utterance) {
    Infer({method: 'enumerate'},
          function() {
              var world = worldPrior();
              var affect = affectPrior(world);
              var goal = goalPrior();
              factor(L1Speaker(world, affect, goal).score(utterance))
              return {world:world, affect:affect};
          }
    );
};
//Visualization
// viz.auto(Infer({method:'enumerate'}, worldPrior));
viz.marginals(L1Listener("all"));
~~~~
