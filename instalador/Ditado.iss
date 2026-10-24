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
var
  PaginaTecla: TInputOptionWizardPage;

function ConfigDoUsuario: String;
begin
  // /CONFIGDIR=<pasta> existe so para testes do instalador; normalmente e %APPDATA%\Ditado
  Result := ExpandConstant('{param:CONFIGDIR|{userappdata}\Ditado}') + '\config.toml';
end;

// Valor atual de "tecla" na config existente ('' se nao houver)
function TeclaDaConfig: String;
var
  Linhas: TArrayOfString;
  i, p: Integer;
  L: String;
begin
  Result := '';
  if not LoadStringsFromFile(ConfigDoUsuario, Linhas) then Exit;
  for i := 0 to GetArrayLength(Linhas) - 1 do
  begin
    L := Trim(Linhas[i]);
    if (Pos('tecla', L) = 1) and (Pos('=', L) > 0) then
    begin
      if Pos('ctrl_l', L) > 0 then Result := 'ctrl_l'
      else if Pos('ctrl_r', L) > 0 then Result := 'ctrl_r';
      Exit;
    end;
  end;
end;

// Escolha final: parametro /TECLA=ctrl_l|ctrl_r (instalacao silenciosa) ou o que foi marcado na pagina
function TeclaEscolhida: String;
var
  P: String;
begin
  P := ExpandConstant('{param:TECLA|}');
  if (P = 'ctrl_l') or (P = 'ctrl_r') then Result := P
  else if (PaginaTecla <> nil) and PaginaTecla.Values[0] then Result := 'ctrl_l'
  else if (PaginaTecla <> nil) and PaginaTecla.Values[1] then Result := 'ctrl_r'
  else if TeclaDaConfig <> '' then Result := TeclaDaConfig
  else Result := 'ctrl_r';
end;

procedure InitializeWizard;
var
  Atual: String;
begin
  PaginaTecla := CreateInputOptionPage(wpSelectTasks, 'Tecla para ditar',
    'Qual tecla você vai segurar para falar?',
    'Segure a tecla, fale e solte: o texto aparece onde o cursor estiver. Você pode mudar depois pelo ícone do Ditado, perto do relógio.',
    True, False);
  PaginaTecla.Add('Ctrl esquerdo');
  PaginaTecla.Add('Ctrl direito (padrão)');
  Atual := ExpandConstant('{param:TECLA|}');
  if Atual = '' then Atual := TeclaDaConfig;
  PaginaTecla.Values[0] := (Atual = 'ctrl_l');
  PaginaTecla.Values[1] := (Atual <> 'ctrl_l');
end;

// Grava a tecla na config do usuario: cria a partir do modelo ou troca so a linha "tecla", preservando o resto
procedure GravarTecla;
var
  Linhas: TArrayOfString;
  i: Integer;
  Achou: Boolean;
  Alvo: String;
begin
  Alvo := ConfigDoUsuario;
  ForceDirectories(ExtractFileDir(Alvo));
  if not FileExists(Alvo) then
    FileCopy(ExpandConstant('{app}\_internal\config.padrao.toml'), Alvo, False);
  if not LoadStringsFromFile(Alvo, Linhas) then Exit;
  Achou := False;
  for i := 0 to GetArrayLength(Linhas) - 1 do
    if (not Achou) and (Pos('tecla', Trim(Linhas[i])) = 1) and (Pos('=', Linhas[i]) > 0) then
    begin
      Linhas[i] := 'tecla = "' + TeclaEscolhida + '"';
      Achou := True;
    end;
  if Achou then SaveStringsToUTF8File(Alvo, Linhas, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then GravarTecla;
end;

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
