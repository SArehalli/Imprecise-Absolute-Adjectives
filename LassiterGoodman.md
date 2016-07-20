Continuous, KDE based Lassiter and Goodman 2013. Super slow and inaccurate, ***do not use***.

~~~~
var COST = 0.6
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
var COST = 2/3
var RATIONALITY = 5
var prec = 500

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

Kao et. al. Modified to be about absolute adjective interpretation.

~~~~
var COST = 2/3
var RATIONALITY = 4
var prec = 500

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
  // This ones a bit weird - ignore endpoints w/ prob 0
  return Discrete({ps:map(function(k) {Math.exp(Beta({a:a, b:b}).score((k+1)/(N+2)))}, 
                          genArray(N))});
}

// Priors
var worldPrior = function() {
    return (sample(DiscreteBeta(prec, 1, 1)))/prec;
};

var thresholdPrior = function() {
    return (sample(DiscreteUniform(prec)))/prec;
};

var utterancePrior = function() {
    var lexicon = ["", "safe", "dangerous"];
    return lexicon[randomInteger(lexicon.length)];
};

var granularityPrior = function() {
    return (1/Math.pow(2, sample(DiscreteUniform(4)) + 1));
}


// Literal meaning and cost - threshold fixed at max
var semantics = function(u, w) {
    return u == "dangerous" ? w == 1 : 
           u == "safe"? w < 1 :
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
              var utility = Math.log(sum(map(function(w) {Math.exp(LiteralListener(utterance).score(w))},
                                    filter(function(w) {Math.abs(world - w) < granularity}, map(function(k) {k/prec}, 
                                                                                                genArray(prec))))));
              factor(RATIONALITY * (utility - cost(utterance)));
              return utterance;
          }
    );
});

var L1Listener = function(utterance, granularity) {
    Infer({method: 'enumerate'},
          function() {
              var world = worldPrior();
              factor(L1Speaker(world, granularity).score(utterance));
              return {world:world, granularity:granularity};
          }
    );
};

var L2Speaker = cache(function(world, granularity) {
    Infer({method: 'enumerate'},
          function() {
              var utterance = utterancePrior();
              var utility = L1Listener(utterance, granularity).score(w);                           
              factor(RATIONALITY * (utility - cost(utterance)));
              return utterance;
          }
    );
});

var L2Listener = function(utterance) {
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
viz.marginals(L2Listener("safe"));
~~~~
