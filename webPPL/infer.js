var actions = ['A', 'B'];
var startStates = ['start1','start2']


var transition = function(state, action) {
  var nextStates = ['A', 'B'];
  var nextProbs = (action === 'A') ? [1,0] : [0,1];
  return categorical(nextProbs, nextStates);
};


var expectedUtility = function(state, action, utility) {
                return utility(transition(state, action));
          };


var softMaxAgent = function(state, beta, utility) {
      return Infer({ 
        model() {
          var action = uniformDraw(actions);
          var eu = expectedUtility(state, action, utility);
          print("action, state, beta, eu =");
          print(action);
          print(state);
          print(beta);
          print(eu);
          factor(eu/beta);
          
        return action;
        }
      });

};

var makeTrajectory = function(getBeta, utility, length) {
  var step = function(){
        var state = uniformDraw(startStates);
        var action = sample(softMaxAgent(state, getBeta(state), utility));
        return {state, action};
  };
  var res = step()
  return length==0 ? [] : [res].concat(makeTrajectory(getBeta, utility, length-1));
};

var observedTrajectory = [['start2','A'], ['start2','A'], ['start2','A']];

var posterior = dp.cache(function(observedTrajectory){
  return Infer({model() {
  //define priors
  var beta1 = sample(RandomInteger({n:8}))+1;
  var beta2 = sample(RandomInteger({n:8}))+1;
  
  var getBeta = function(state){
    var table = {
      start1: beta1,
      start2: beta2
    };
    return table[state];
  };
  
  var d = sample(RandomInteger({n:9}))-4
  var utilA = 0;
  var utilB = utilA + d;
  
  var utility = function(state){
    var table = {
      A: utilA,
      B: utilB
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

  //var choice = d<0 ? 'A' : 'B' //choose the action with higher utility
  return {d};
  }});
});

var score = Infer( {model() {
  //define true values
  var trueBeta1 = sample(RandomInteger({n:8}))+1;
  var trueBeta2 = sample(RandomInteger({n:8}))+1;
  
  var getTrueBeta = function(state){
    var table = {
      start1: trueBeta1,
      start2: trueBeta2
    };
    return table[state];
  };
  
  var trueD = sample(RandomInteger({n:9}))-4
  var trueUtilA = 20;
  var trueUtilB = trueUtilA + trueD;
  
  var trueUtility = function(state){
    var table = {
      A: trueUtilA,
      B: trueUtilB
    };
    return table[state];
  };
  
  var observedTrajectory = makeTrajectory(getTrueBeta, trueUtility, 10);
  
  var d = sample(posterior(observedTrajectory))
  var sameSign = d*trueD >0; //true if d and trueD have same sign
  var reward = sameSign ? 
      expectation(Math.abs(trueD)) 
   : -expectation(Math.abs(trueD))
  
}});
