--' minutes
SELECT DISTINCT Season, GameDate, box.NCAAGameId AS GameId, 
NCAATeamId AS TeamId, NCAAPlayerId AS PlayerId, 
(40.0 * mp/(40.0 + 5.0 * overtimeperiods)) AS Mins
FROM Import_NCAA_PlayerGameStats box
JOIN Import_NCAA_Games game ON box.`NCAAGameId` = game.`NCAAGameId`
WHERE NCAATeamId IN 
(SELECT NCAATeamId FROM Map_NCAA_Synergy_Teams ncaa
JOIN Import_Synergy_Teams syn ON ncaa.`SynergyTeamId` = syn.`SynergyTeamId`
WHERE `SynergyDivision` = 'NCAA Division I')
AND `OppNCAATeamId` IN 
(SELECT NCAATeamId FROM Map_NCAA_Synergy_Teams ncaa
JOIN Import_Synergy_Teams syn ON ncaa.`SynergyTeamId` = syn.`SynergyTeamId`
WHERE `SynergyDivision` = 'NCAA Division I') {season}
ORDER BY Season, GameDate;

--' injury
SELECT NCAATeamId AS TeamId, NCAAPlayerId AS PlayerId, 
Date(Date) AS Date, InjuryStatus
FROM Proc_Injury_Players;

--' exempt
SELECT NCAAPlayerId AS PlayerId, ExemptDate
FROM Manual_NCAA_Exempt_Player
WHERE Ineligible = 1;
