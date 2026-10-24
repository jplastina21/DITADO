; Instalador do Ditado (Inno Setup 6). Gera Output\Ditado-Setup.exe a partir de dist\Ditado\.
#define AppNome "Ditado"
#define AppVersao "1.0.0"

[Setup]
AppId={{6E5C1A3E-0B6D-4C57-9B0A-D17AD0C0DE01}
AppName={#AppNome}
AppVersion={#AppVersao}
AppPublisher=Ditado
DefaultDirName={autopf}\{#AppNome}
DefaultGroupName={#AppNome}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=Ditado-Setup
SetupIconFile=ditado.ico
UninstallDisplayIcon={app}\Ditado.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
AppMutex=DitadoInstaladoMutex
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar um atalho na Área de Trabalho"; GroupDescription: "Atalhos:"
Name: "startup"; Description: "Abrir o Ditado junto com o Windows"; GroupDescription: "Início:"

[Files]
Source: "dist\Ditado\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Ditado"; Filename: "{app}\Ditado.exe"; Comment: "Ditado por voz (segure Ctrl direito e fale)"
Name: "{group}\Desinstalar o Ditado"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Ditado"; Filename: "{app}\Ditado.exe"; Tasks: desktopicon; Comment: "Ditado por voz (segure Ctrl direito e fale)"

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Ditado"; ValueData: """{app}\Ditado.exe"""; Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\Ditado.exe"; Description: "Abrir o Ditado agora"; Flags: nowait postinstall skipifsilent

[Code]
// Ao desinstalar, pergunta se apaga tambem a configuracao e os modelos baixados (ficam fora da pasta do programa).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and (not UninstallSilent) then
    if MsgBox('Apagar também sua configuração, o log e os modelos de voz já baixados (podem ocupar mais de 1 GB)?',
              mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    begin
      DelTree(ExpandConstant('{userappdata}\Ditado'), True, True, True);
      DelTree(ExpandConstant('{localappdata}\Ditado'), True, True, True);
    end;
end;
