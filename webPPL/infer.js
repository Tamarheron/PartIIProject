var max = function(a,b){
  return Math.max(a,b);
};


var actions = ['A', 'B'];
var startStates = ['start1','start2'];
var nextStates = ['A', 'B'];


var transition = function(state, action) {
  var nextProbs = (action === 'A') ? [1,0] : [0,1];
  return categorical(nextProbs, nextStates);
};

//distributions of true parameters and robot's belief
var TRUEBETA1 = Categorical({ps:[1,1,1], vs:[1,2,3]});
var TRUEBETA2 = Categorical({ps:[1,1,1], vs:[4,2,3]});
var TRUED = Categorical({ps:[1,1], vs:[-1,1]});

var expectedUtility = function(state, action, utility) {
                return utility(transition(state, action));
          };


var softMaxAgent = function(state, beta, utility) {
      return Infer({ 
        model() {
          var action = uniformDraw(actions);
          var eu = expectedUtility(state, action, utility);
          var debugPrint = false;
          if (debugPrint){
            print("action, state, beta, eu =");
            print(action);
            print(state);
            print(beta);
            print(eu);
            print("factor");
            print(eu/beta);
          }
          
          factor(eu/beta);
          
        return action;
        }
      });

};
//generate sample trajectory
var makeTrajectory = function(getBeta, utility, length) {
  var step = function(){
    var state = uniformDraw(startStates);
    var action = sample(softMaxAgent(state, getBeta(state), utility));
    return [state, action];
  };
  var res = step()
  return length==1 ? [res] : [res].concat(makeTrajectory(getBeta, utility, length-1));
};


var posterior = dp.cache(function(observedTrajectory){
  return Infer({model() {
  //define priors
  var beta1 = sample(TRUEBETA1);
  var beta2 = sample(TRUEBETA2);
  
  var getBeta = function(state){
    var table = {
      start1: beta1,
      start2: beta2
    };
    return table[state];
  };
  
  var d = sample(TRUED);
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
    return {d};
  }});
});

//model the regret of the human based on their beta level
var humanScore = function(state) {
  Infer( {model(){
  //define true values
  var trueBeta1 = sample(TRUEBETA1);
  var trueBeta2 = sample(TRUEBETA2);
  
  var getTrueBeta = function(state){
    var table = {
      start1: trueBeta1,
      start2: trueBeta2
    };
    return table[state];
  };
  
  var trueD = sample(TRUED);
  var trueUtilA = 20;
  var trueUtilB = trueUtilA + trueD;
  
  var trueUtility = function(state){
    var table = {
      A: trueUtilA,
      B: trueUtilB
    };
    return table[state];
  };
    
    var action = uniformDraw(actions);
    var eu = expectedUtility(state, action, trueUtility);
    var beta = getTrueBeta(state);
    //print("state, action, eu, beta=");
    //print(state);
    //print(action);
    //print(eu);
    //print(beta);
    factor(eu/beta);
    var actualUtility = trueUtility(transition(action,state));
    var utilities = map(trueUtility, nextStates);
    var maxUtility = reduce(max, -999999, utilities);
    var regret = actualUtility - maxUtility;
    //print("actual, max, regret=");
    //print(actualUtility);
    //print(maxUtility);
    //print(regret);
    return {regret, state};
   }});
};


var score = function(length){
  Infer( {model() {
  //define true values
  var trueBeta1 = sample(TRUEBETA1);
  var trueBeta2 = sample(TRUEBETA2);
  
  var getTrueBeta = function(state){
    var table = {
      start1: trueBeta1,
      start2: trueBeta2
    };
    return table[state];
  };
  
  var trueD = sample(TRUED);
  var trueUtilA = 20;
  var trueUtilB = trueUtilA + trueD;
  
  var trueUtility = function(state){
    var table = {
      A: trueUtilA,
      B: trueUtilB
    };
    return table[state];
  };
  
  var observedTrajectory = makeTrajectory(getTrueBeta, trueUtility, length);
  var d = sample(posterior(observedTrajectory)).d
  var correctChoice = d*trueD >0; //true if d and trueD have same sign
  var regret = correctChoice ? 0 : -Math.abs(trueD)
  return {regret,length};
  
}});
};
var start = Categorical({ps:[1,1], vs:startStates});
//var observedTrajectory1 = [['start1','A'],['start1','A'],['start2','A'],['start2','A']];
//var observedTrajectory2 = [['start1','A']];
//viz(posterior(observedTrajectory1));
//viz(posterior(observedTrajectory2));
//var robotRegrets = map(score, [1,5,10,15,20]);
//map(viz, robotRegrets);
var humanRegret = map(humanScore, ['start1', 'start2']);
map(viz,humanRegret);