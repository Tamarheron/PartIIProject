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
      return Infer({ method:"enumerate",
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


var posterior = function(observedTrajectory){
  return Infer({method:"enumerate", model(){
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
  return d;
  }});
};

//model the regret of the human based on their beta level
var humanScore = function(state) {
  Infer( {method:"enumerate", model(){
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
    return regret;
   }});
};


var robotScore = function(length){
  Infer( { model() {
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
  var posteriorVar = posterior(observedTrajectory);
  //print(posteriorVar);
  var d = expectation(posteriorVar);
  //print(d);
  var correctChoice = d*trueD >0; //true if d and trueD have same sign
  var regret = correctChoice ? 0 : -Math.abs(trueD)
  return {regret,length};
  
}});
};

var naiveRobotScore = function(length){
  Infer({model() {
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
    
  //function to count which is the most frequent action
  var getCounts = function(traj) {
    var count = function(target){
      var test = function(stateActionPair) {
        return (target == stateActionPair[1]) ? 1 : 0
      }
      var flags = map(test, traj); //array containing a 1 for each match
      return sum(flags);
    }
    var counts = map(count, actions);
    //print(counts);
    return counts;
  }
  var observedTrajectory = makeTrajectory(getTrueBeta, trueUtility, length);
  var posteriorVar = posterior(observedTrajectory);
  var counts = getCounts(observedTrajectory);
  var d = counts[1]-counts[0];
  if (d==0) {
    //in this case we choose at random
    var regret = -0.5*Math.abs(trueD);
    //print(regret);
    return {regret,length};
  } 
  else{
    var correctChoice = d*trueD >0; //true (we choose correctly) if d and trueD have same sign
    var regret = correctChoice ? 0 : -Math.abs(trueD)
    //print(regret);
    return {regret,length};
  }
    
  }});
}
//var observedTrajectory1 = [['start1','A'],['start1','A'],['start2','A'],['start2','A']];
//var observedTrajectory2 = [['start1','A']];
//var post1 = posterior(observedTrajectory1);
//var post2 = posterior(observedTrajectory2);
//viz(post1);
//viz(post2);
//print(expectation(post1));

var robotRegrets = map(robotScore, [1,3,5,9]);
print(robotRegrets);

//var humanRegrets = map(humanScore, ['start1', 'start2']);
//print(humanRegrets);
//map(viz,humanRegrets);
//var humanMeans = map(expectation, humanRegrets);
//var humanRegret = listMean(humanMeans);
//print(humanRegret);
  
var naiveRobotRegrets = map(naiveRobotScore, [1,3,5,9]);
print("naive");
print(naiveRobotRegrets);


map(viz, robotRegrets);
map(viz, naiveRobotRegrets);
