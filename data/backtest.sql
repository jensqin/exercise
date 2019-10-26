SELECT Version, Season, Week, GameId, FranchiseId, OpponentId, PlayerId, 
Unit, GamePosition, Role, Ratio, Status, AdjPred
FROM Model_NFL_RoleSnap
WHERE Season = 2019 AND Version IN (0, 1);
