var actions = ['A', 'B'];
var startStates = ['start1','start2']


var transition = function(state, action) {
  var nextStates = ['A', 'B'];
  var nextProbs = (action === 'A') ? [1,0] : [0,1];
  return categorical(nextProbs, nextStates);
};

var utility = function(state) {
  var table = { 
    A: 10,
    B: 5,
    C: 1
  };
  return table[state];
};


var expectedUtility = function(state, action) {
            return expectation(Infer({ 
              model() {
                return utility(transition(state, action));
              }
            }));
          };


var softMaxAgent = function(state, beta, utility) {
      return Infer({ 
        model() {
          var action = uniformDraw(actions);
          var eu = expectedUtility(state, action);
          factor(eu/beta);
          
        return action;
        }
      });

};

var observedTrajectory = [['start2','B'], ['start2','B'], 
                          ['start2','B'], ['start2','A'], 
                          ['start2','A']];


var posterior = Infer({model() {
  var beta1 = sample(RandomInteger({n:10}))+1;
  var beta2 = sample(RandomInteger({n:10}))+5;

  var getBeta = function(state){
    var table = {
      start1: beta1,
      start2: beta2
    };
    return table[state];
  };
  // For each observed state-action pair, factor on likelihood of action
  map(
    function(stateAction){
      var state = stateAction[0];
      var beta = getBeta(state);
      var action = stateAction[1];
      observe(softMaxAgent(state, beta, utility), action);
    },
    observedTrajectory);
  
  //var beta2=betas.beta2;
  return { beta2 };
}});

viz(posterior);