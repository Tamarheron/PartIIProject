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
    B: 5
  };
  return table[state];
};

var getBeta = function(state){
  var table = {
    start1: 1,
    start2: 20
  };
  return table[state];
};


var softMaxAgent = function(state) {
  return Infer({ 
    model() {

      var action = uniformDraw(actions);
      var beta = getBeta(state);
      var expectedUtility = function(action) {
        return expectation(Infer({ 
          model() {
            return utility(transition(state, action));
          }
        }));
      };
      
      factor(expectedUtility(action)/beta);
      
      return action;
    }
  });
};

var startState = uniformDraw(startStates);

viz(softMaxAgent(startState));