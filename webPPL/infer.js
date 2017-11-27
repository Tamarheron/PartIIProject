var max = function(a,b){
  return Math.max(a,b);
};

var createMDP = function(options){
  // takes parameters: states, startStates, actions, transitions, utilities, betas
  assert.ok(_.has(options, 'states') &&
            _.has(options, 'startStates') &&
            _.has(options, 'actions') &&
            _.has(options, 'transitions') &&
            _.has(options, 'utilities') &&
            _.has(options, 'betas'),
      'createMDP args: options does not contain one or more' +
            ' of states, startStates, actions, transitions, utilities, betas');
  var states = options.states;
  
  var getBeta = function(state){
    var index = states.indexOf(state);
    assert.ok((index!=-1), 'state not found');
    return options.betas[index];
  };
  
  var getActions = function(state){
    var index = options.states.indexOf(state);
    assert.ok((index!=-1), 'state '+state+' not found');
    return options.actions[index];
  };
  
  var getUtility = function(state){
    var index = options.states.indexOf(state);
    assert.ok((index!=-1), 'state not found');
    return options.utilities[index];
  };
  var transitions = options.transitions;
  var getTransition = options.transitions;
  var startStates = options.startStates;
  var length = options.length;
  
  var getExpectedUtility = function(options) {
       assert.ok(_.has(options, 'state') &&
            _.has(options, 'action'), 
                 'getExpectedUtility args:' +
                 'one or more of state, action is missing');
    var expected =  expectation(Infer({model() { 
         var current = options.state;
         var action = options.action;
         var next = sample(getTransition({state:current, action:action}));
         var util = getUtility(next);
         return util;

       }}));     
    return expected;
    

  };
  
  return{getBeta, getActions, getUtility, getExpectedUtility,
         getTransition, startStates, states, actions, transitions,
        utilities, betas};
};

  


var softMaxAgent = function(options) {
  assert.ok( _.has(options, 'MDP') &&
            _.has(options, 'discount'),
                 'softMaxAgent args:' +
                 'one or more of discount, MDP is missing');
  var MDP = options.MDP;
  var getActions = MDP.getActions;
  var getBeta = MDP.getBeta;
  var discount = options.discount;
  var getTransition=MDP.getTransition;
  var getUtility = MDP.getUtility;
  
  var getEU = function(options) {
    //console.log('called eu');
       assert.ok(_.has(options, 'state') &&
                 _.has(options, 'action') &&
                 _.has(options, 'length'), 
                 'getExpectedUtility args:' +
                 'one or more of state, action is missing');
    var expected =  expectation(Infer({method:'enumerate', model() { 
       var current = options.state;
       var action = options.action;
       var next = sample(getTransition({state:current, action:action}));
       var util = getUtility(next);
       if (length==0){
         return util;
       } else{
         var nextAction = act({state:next, length:length});
         var newLength = length-1;
         var futureUtil = discount*getEU({
           state:next, action:nextAction, length:newLength})
         return (util + futureUtil);

       }}}));  
    return expected;
  };
  
  var act = function(options){
    //console.log('called act');
    assert.ok(_.has(options, 'state') &&
              _.has(options, 'length'), 
                'act args: missing one or more of: state, length');
    var state=options.state;
    var length=options.length;
    var beta = getBeta(state);
    var actions = getActions(state);
    return Infer({ method:"enumerate",
        model() {
          var action = uniformDraw(actions);
          var eu = getEU({state: state, action:action, length:length});
          var debugPrint = false;
          if (debugPrint){
            console.log("action: " + action + " state: " + state + 
                 " beta: " + beta + " eu: "+eu+" factor: "+eu/beta );
          }
          factor(eu/beta);
          
        return action;
        }
   });
  };
  return {getEU, act};
};

//generate sample trajectory
var makeTrajectory = function(options) {
  //console.log('calling maketraj');
   assert.ok(_.has(options, 'length') &&
             _.has(options, 'agent') &&
            _.has(options, 'MDP'), 
                 'makeTrajectory args:' +
                 'one or more of length, agent, MDP is missing');
  var MDP = options.MDP;
  var length = options.length;
  var start = uniformDraw(MDP.startStates);
  //console.log('start is ' + start + ' length is '+ length);
  var agent = options.agent;
  var act=agent.act;
  var transition=MDP.getTransition;
  
  var step = function(options){
    //console.log('calling step');
    assert.ok(_.has(options, 'state') &&
              _.has(options, 'length'), 
                 'makeTrajectory.step args:' +
                 'one or more of length, state is missing');
    var state=options.state;
    var length=options.length;
    var action = sample(act({state:state, length:length}));
    var transDist = transition({state:state, action:action})
    var next = sample(transDist);
    var res = [state, action];
    var newLength = length-1;
    var debug=false;
    if (debug) {console.log('action: ' + action + ' trnsDist: '+transDist+
               ' length: '+length +' newLen: '+newLength+ ' next: '+ next);};
    return length==1 ? 
    [res] : 
    [res].concat(step({state:next, length: newLength}));
  };
  
  return step({state:start, length:length});
  
};


var posterior = function(options){
  assert.ok(_.has(options, 'priors') &&
            _.has(options, 'MDPParams') &&
            _.has(options, 'discount') &&
            _.has(options, 'beta') &&
            _.has(options, 'observedTrajectory'),
           'posterior args missing one or more of ' +
            'priors, MDPParams,discount, beta, observedTrajectory');
  
  var betaType=options.beta;
  //assert.ok((betaType=='fixed') || (betaType=='variable'), 'beta must be "fixed"+
  //          ' or "variable"');
  
  var priors = options.priors;
  var traj = options.observedTrajectory;
  var params = options.MDPParams;
  var discount=options.discount;
  
  var trajLength = traj.length;
  var remainingLengths = mapN(
      function(n){return trajLength-n}, trajLength)
  
  var debug =false;
  
  return Infer({method:"enumerate",model(){
    //sample from priors
    var utilities = map(sample,priors.utilities);
    if (betaType=='variable'){
    var betas = map(sample,priors.betas);
    } else if (betaType=='fixed'){
      var beta = sample(priors.betas)
      var betas = mapN(function(n) {return beta}, utilities.length);
    } 
    
    if (debug) {console.log('sample utilities '+utilities+', betas: '+betas)}
    
    //create MDP based on known params and sampled utilities and betas
    var MDP = createMDP({actions:params.actions, transitions:params.transitions, 
                     states:params.states, startStates:params.startStates, 
                     utilities:utilities, betas:betas});
    var agent = softMaxAgent({MDP:MDP, discount:discount});
    var act = agent.act;

    
    // For each observed state-action pair, factor on likelihood of action,
    //given the remaining length
    map2(
      function(stateAction, length){
        var state = stateAction[0];
        var action = stateAction[1];
        observe(act({state:state,length:length }), action);
      },
      traj, remainingLengths);
    return {utilities:utilities, betas:betas};
  }});
};

var getTotalUtility = function(options){
  assert.ok(_.has(options, 'traj') &&
           _.has(options, 'getUtility'),
           'getTotalUtility args missing one or more of traj, getUtility') 
  var getUtility = options.getUtility;
  var utilities = map(
      function(stateaction){
        var state = stateaction[0];
        return getUtility(state);
      }, options.traj);
    
    var total = sum(utilities);
    return total;
}

//calculate average utility achieved by human
var humanScore = function(options) {
  assert.ok(_.has(options, 'MDP') &&
           _.has(options, 'length') &&
           _.has(options, 'discount'), 'humanScore args: missing one or more'+
           'of MDP, length, discount');
  
  var getUtility = options.MDP.getUtility;
  var agent = softMaxAgent({MDP:options.MDP, discount:options.discount});
  
  //calculate average utility achieved by human
  return expectation(Infer( { model(){
    var traj = makeTrajectory({MDP:options.MDP, length:options.length,
                              agent:agent});
    var utility = getTotalUtility({getUtility:getUtility, traj:traj});
    return utility;
   }}));
};


//function to count which is the most frequent action for each state
var getCounts = function(options) {
  assert.ok(_.has(options, 'traj') &&
           _.has(options, 'states') &&
           _.has(options, 'getActions'), 'getCounts args: missing one or more '+
           'of traj, states, getActions');
  
  var getActions = options.getActions;
  
   var count = function(state, action){
      var test = function(stateActionPair) {
        return ((state==stateActionPair[0])&&(action==stateActionPair[1])) ? 1 : 0
      }
      var flags = map(test, options.traj); //array containing a 1 for each match
      return sum(flags);
   }
   
   //for each state, list for each available action the counts for that action
   var counts = 
       map(
         function(state){
           return map(
             function(action){
               return count(state, action);
             }, 
             getActions(state)
           )},
         options.states
       );
  
    //print(counts);
    return counts;
};
 

var naiveRobotScore = function(options){
  //Naive robot chooses the most frequent action taken in each state
  //in the observed traj
  var debug1=false;
  var debug2=false;
  
  assert.ok(_.has(options, 'MDP') &&
           _.has(options, 'length') &&
           _.has(options, 'discount'), 'naiveRobot Score args: missing one '+
           'or more of MDP, length, discount');
  
  var MDP=options.MDP;
  var getUtility = MDP.getUtility;
  var getActions = MDP.getActions;
  var states = MDP.states;
  var agent = softMaxAgent({MDP:MDP, discount:options.discount});
  var length = options.length;
  
  var robotAction = function(state, counts){
    var index = states.indexOf(state);
    assert.ok((index!=-1), 'robotAction: state not found');
    var actionCounts = counts[index];
    var maxActionCount = reduce(max, -9999999, actionCounts);
    
    //if multiple actions are most frequent, randomly return first or last one
    var mostFrequentAction1 = actionCounts.indexOf(maxActionCount);
    var mostFrequentAction2 = actionCounts.lastIndexOf(maxActionCount);
    var actionIndex = uniformDraw([mostFrequentAction1, mostFrequentAction2]);
    var actions = getActions(state);
    return actions[actionIndex];
  }
  
  return expectation(Infer({ model() {  
    var observedTrajectory = makeTrajectory({MDP:MDP, length:length, agent:agent}); 
    var counts = getCounts({traj:observedTrajectory, states:states,
                            getActions:getActions});
    
    
   
    var start = uniformDraw(MDP.startStates);
    
    var step = function(options){
      assert.ok(_.has(options, 'state') &&
                _.has(options, 'length'), 
                'makeTrajectory.step args:' +
                'one or more of length, state is missing');
      var state=options.state;
      var length=options.length;
      var action = robotAction(state, counts);
      var transDist = transition({state:state, action:action})
      var next = sample(transDist);
      var res = [state, action];
      var newLength = length-1;
      
      if (debug1) {console.log('action: ' + action + ' transDist: '+transDist+
                              ' length: '+length +' newLen: '+newLength+
                              ' next: '+ next + ' counts: '+counts);};
      return length==1 ? 
        [res] : 
      [res].concat(step({state:next, length: newLength}));
    };
  
   var robotTraj = step({state:start, length:length}); 
    
   if (debug2) {console.log('obs, counts, robot ' +
                           observedTrajectory, counts, robotTraj);;}
   return getTotalUtility({traj:robotTraj, getUtility:getUtility});
  }}));
};


var IRLRobotScore = function(options){
   assert.ok(_.has(options, 'MDP') &&
             _.has(options, 'length') &&
             _.has(options, 'priors') &&
             _.has(options, 'beta') &&
             _.has(options, 'discount'), 'IRLRobotScore args: missing one '+
             'or more of MDP, beta, length, priors, discount');
  
  var betaType=options.beta;
  //assert.ok((betaType=='fixed')||(betaType=='variable'), 'beta must be "fixed"+
  //          ' or "variable"');
  
  var MDP=options.MDP;
  var human = softMaxAgent({MDP:MDP, discount:options.discount});
  var length = options.length;
  var MDPParams = ({actions:MDP.actions, transitions:MDP.transitions, 
                     states:MDP.states, startStates:MDP.startStates});
  var getUtility = MDP.getUtility;
  var robotBetas = mapN(function(n){return 0.001}, length); //robot has very low beta
  
  return expectation(Infer({model(){
    var observedTrajectory = makeTrajectory({MDP:MDP, length:length, agent:human}); 
    var sample = sample(posterior({priors:options.priors, 
                                   discount:options.discount,
                                   observedTrajectory:observedTrajectory,
                                   MDPParams:MDPParams,
                                   beta:betaType}));
    
   
    //robot MDP has sampled utilities and very low betas everywhere
    var robotMDP = createMDP({actions:params.actions, transitions:params.transitions, 
                     states:params.states, startStates:params.startStates, 
                     utilities:sample.utilities, betas:robotBetas});
    var robot = softMaxAgent({MDP:robotMDP, discount:options.discount});
    var robotTraj = makeTrajectory({length:length, MDP:robotMDP, 
                               agent:robot});
    
    return getTotalUtility({traj:robotTraj, getUtility:getUtility});

  }}));
};


var actions = [['A', 'B'],['A', 'B'], ['A', 'B'], ['A', 'B']];
var transition = function(options) {
  assert.ok( _.has(options, 'state') && 
             _.has(options, 'action'), 'transition args: missing one or more ' +
           'of state, action');
  var action=options.action;
  var state=options.state;
  var nextProbs = (action === 'A') ? [0,0,1,0] : [0,0,0,1];
  return Categorical({ps:nextProbs, vs:states});
};

var states = ['start1', 'start2', 'A', 'B'];
var startStates = ['start1','start2'];
var betas = [1,1,1,10];
var utilities = [1,1,1,10];

var params = {actions:actions, transitions:transition, states:states, 
              startStates:startStates};

var utilityPriors = mapN(
  function(n){return Categorical({ps:[1,1], vs:[1,10]})}
  , 4);
var betaPriors = mapN(
  function(n){Categorical({ps:[1,1], vs:[1,10]})}, 
  4);
var singleBetaPrior = Categorical({ps:[11], vs:[1,10]});
var priors1 = {utilities:utilityPriors, betas:betaPriors};
var priors2 = {utilities:utilityPriors, betas:singleBetaPrior};

console.log(utilityPriors, betaPriors);
console.log('call 3');
var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                    transitions:transition, utilities:utilities, betas:betas});

var agent = softMaxAgent({discount:0.9, MDP:MDP});
//var traj = makeTrajectory({length:5, agent:agent, MDP:MDP});

//print(traj);

//var post = posterior({priors:priors, MDPParams:params, 
//                      discount:0.9, observedTrajectory:traj});
//viz(post);
var params = {MDP:MDP, discount:0.9, length:8}
var score = humanScore(params);
print('human '+ score);
//var getActions = MDP.getActions;
//print(actions)
//print(states);
//print(map(getActions, states));
//var counts = getCounts({states:states, getActions:getActions, traj:traj});
//print(counts);

var params1 = {MDP:MDP, discount:0.9, length:8, priors:priors2, beta:"fixed"};
var params2 = {MDP:MDP, discount:0.9, length:8, priors:priors1, beta:"variable"};

var robotScoreFixed = IRLRobotScore(params1);
print('fixed '+robotScoreFixed);

var robotScoreVariable = IRLRobotScore(params2);
print('variable '+robotScoreVariable);