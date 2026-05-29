# configure_scheduler.ps1
# Replaces the old "repeat every 1 hour for 9 hours" task with 4 targeted triggers.
# Run once as Administrator:  .\configure_scheduler.ps1
# Times are local Mountain Time (MST/MDT) — Windows uses the machine's local clock.

$TaskName   = "Baseball Model"
$TaskPath   = "\"               # root task folder
$ScriptDir  = $PSScriptRoot     # resolves to the folder containing this script
$BatFile    = "$ScriptDir\run_model.bat"

# --- Remove any existing tasks with the old or new name ---
foreach ($name in @("Baseball Model 9am", "Baseball Model")) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed old task: $name"
    }
}

# --- Action: run_model.bat (log redirects are inside the bat file; do NOT add >> here) ---
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`""

# --- Principal: run as current user, highest privileges ---
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Password `
    -RunLevel Highest

# --- Settings ---
$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# --- 4 Daily Triggers (Mountain Time) ---
# Run 1 — 6:00 PM MST  : opening lines drop for next day
# Run 2 — 11:00 PM MST : late-posted / west-coast lines
# Run 3 — 5:00 AM MST  : morning board catch
# Run 4 — 11:00 AM MST : final pre-slate lineup/pitcher verification

$Triggers = @(
    $(New-ScheduledTaskTrigger -Daily -At "18:00"),
    $(New-ScheduledTaskTrigger -Daily -At "23:00"),
    $(New-ScheduledTaskTrigger -Daily -At "05:00"),
    $(New-ScheduledTaskTrigger -Daily -At "11:00")
)

# Register with all 4 triggers
$Task = Register-ScheduledTask `
    -TaskName   $TaskName `
    -Action     $Action `
    -Principal  $Principal `
    -Settings   $Settings `
    -Trigger    $Triggers[0] `
    -Force

# Add remaining triggers (schtasks only accepts one via Register; add the rest via COM)
$SvcObj   = New-Object -ComObject Schedule.Service
$SvcObj.Connect()
$TaskDef  = $SvcObj.GetFolder("\").GetTask($TaskName).Definition

foreach ($t in $Triggers[1..3]) {
    # Convert PowerShell trigger to COM trigger (TASK_TRIGGER_DAILY = 2)
    $ComTrig = $TaskDef.Triggers.Create(2)
    $ComTrig.StartBoundary = $t.StartBoundary   # ISO 8601 datetime string
    $ComTrig.DaysInterval  = 1
    $ComTrig.Enabled       = $true
}
$SvcObj.GetFolder("\").RegisterTaskDefinition(
    $TaskName, $TaskDef, 4,   # 4 = TASK_CREATE_OR_UPDATE
    $env:USERNAME, $null, 1   # 1 = TASK_LOGON_PASSWORD (prompts for password)
) | Out-Null

Write-Host ""
Write-Host "Task '$TaskName' configured with 4 daily triggers (Mountain Time):"
Write-Host "  Run 1 — 6:00 PM  (6PM_Overnight  : opening lines)"
Write-Host "  Run 2 — 11:00 PM (11PM_Overnight : late/west-coast lines)"
Write-Host "  Run 3 — 5:00 AM  (5AM_Morning    : overnight movement)"
Write-Host "  Run 4 — 11:00 AM (11AM_Midday    : pre-slate verification)"
Write-Host ""
Write-Host "NOTE: You will be prompted for your Windows password to enable"
Write-Host "      'run whether logged on or not' mode. This is required for"
Write-Host "      unattended overnight runs while the PC is sleeping."
Write-Host ""
Write-Host "Verify with:  Get-ScheduledTaskInfo -TaskName '$TaskName'"
