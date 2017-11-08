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

var makeBetas = function(){
  var beta1 = sample(RandomInteger({n:10}))+1;
  var beta2 = sample(RandomInteger({n:10})) + 10;

  var getBeta = function(state){
    var table = {
      start1: beta1,
      start2: beta2
    };
    return table[state];
  };

  return{
    beta1, 
    beta2,
    getBeta
  };
};






var softMaxAgent = function(state, beta) {
      var expectedUtility = function(state, action) {
        return expectation(Infer({ 
          model() {
            return utility(transition(state, action));
          }
        }));
      };
      return Infer({ 
        model() {
          var action = uniformDraw(actions);
          var eu = expectedUtility(state, action);
          factor(eu/beta);
          
        return action;
        }
      });
};

var observedTrajectory = [['start1','A']];

var simulate = function(){
  return Infer({ 
      model() {

        var state = uniformDraw(startStates);
        softMaxAgent(state, getBeta(state), utility);
      }
  });
};

var posterior = Infer({ model() {
  var betas = makeBetas();
  var state = uniformDraw(startStates);
  var beta = betas.getBeta(state);
  // For each observed state-action pair, factor on likelihood of action
  map(
    function(stateAction){
      var state = stateAction[0];
      var action = stateAction[1];
      observe(softMaxAgent(state, beta), action);
    },
    observedTrajectory);

  return betas.beta1;
}});

softMaxAgent(start1, getBeta(start1));