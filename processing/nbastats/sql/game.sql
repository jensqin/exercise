SELECT GameId,
    GameDate,
    Season,
    GameType,
    HomeTeamAbbr,
    AwayTeamAbbr,
    HomeTeamId,
    AwayTeamId,
    HomeFinalScore,
    AwayFinalScore,
    NumOTPeriods,
    GameStartEST
FROM Proc_NBAStats_Games;