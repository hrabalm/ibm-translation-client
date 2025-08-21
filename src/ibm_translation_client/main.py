import asyncio
import hashlib
from pathlib import Path

import click

import ibm_translation_client


def document_hash(content: str) -> str:
    document_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return document_hash


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.group()
@click.option(
    "--base-url",
    required=True,
)
@click.option("--token", required=True)
@click.option("--model", default="ibm", show_default=True)
@click.option("--src-lang", default="en", show_default=True)
@click.option("--tgt-lang", default="ja", show_default=True)
@click.option("--glossary-id", default="glos_v1", show_default=True)
@click.option("--dnt-id", default="dnt_v1", show_default=True)
@click.pass_context
def translate(
    ctx,
    base_url,
    token,
    model,
    src_lang,
    tgt_lang,
    glossary_id,
    dnt_id,
):
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url
    ctx.obj["token"] = token
    ctx.obj["model"] = model
    ctx.obj["src_lang"] = src_lang
    ctx.obj["tgt_lang"] = tgt_lang
    ctx.obj["glossary_id"] = glossary_id
    ctx.obj["dnt_id"] = dnt_id


async def translate_file_async(
    client: ibm_translation_client.TranslationClient,
    ctx: click.Context,
    input_file: str,
    output_file: str,
):
    extension = input_file.split(".")[-1].lower()
    content = click.open_file(input_file, "r", encoding="utf-8").read()

    job = ibm_translation_client.TranslationJob(
        model=ctx.obj["model"],
        src_lang=ctx.obj["src_lang"],
        tgt_lang=ctx.obj["tgt_lang"],
        content=content,
        extension=extension,
        glossary_id=ctx.obj["glossary_id"],
        do_not_translate_id=ctx.obj["dnt_id"],
    )
    result = await client.translate_file(job)

    if result is None:
        click.echo(f"FAILURE:\t{input_file}\t{document_hash(job.content)}", err=True)
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with click.open_file(output_file, "w", encoding="utf-8") as f:
            f.write(result.content)
        click.echo(f"SUCCESS:\t{input_file}\t{document_hash(job.content)}", err=True)


@translate.command()
@click.pass_context
@click.argument(
    "input_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        file_okay=True,
        allow_dash=True,
    ),
)
@click.option(
    "--output-file",
    "-o",
    default="-",
    show_default=True,
)
def file(ctx, input_file, output_file):
    client = ibm_translation_client.TranslationClient(
        base_url=ctx.obj["base_url"],
        token=ctx.obj["token"],
        max_concurrent=1,
    )
    asyncio.run(translate_file_async(client, ctx, input_file, output_file))


def map_output_path(src: Path, src_root: Path, dst_root: Path) -> Path:
    """Return the output path that corresponds to `src` under `dst_root`."""
    rel = src.relative_to(src_root)
    return dst_root / rel


@translate.command()
@click.pass_context
@click.argument("input_dir")
@click.option("--output-dir", "-o", required=True)
@click.option(
    "--max-concurrent",
    default=20,
    help="Maximum number of files translated concurrently.",
    show_default=True,
)
@click.option(
    "--extensions",
    "-e",
    default="html,dita,ditamap,svg,xlf",
    show_default=True,
    help="Comma-separated list of file extensions to include.",
)
def batch(ctx, input_dir, output_dir, max_concurrent, extensions):
    included_extensions = set(extensions.split(","))
    input_files = []
    for p in Path(input_dir).rglob("*"):
        if p.is_file() and p.suffix[1:] in included_extensions:
            input_files.append(p)
    click.echo(f"Found {len(input_files)} files to process.", err=True)

    output_files = [
        map_output_path(p, Path(input_dir), Path(output_dir)) for p in input_files
    ]
    client = ibm_translation_client.TranslationClient(
        base_url=ctx.obj["base_url"],
        token=ctx.obj["token"],
        max_concurrent=max_concurrent,
    )

    async def _process():
        jobs = [
            translate_file_async(client, ctx, str(input_file), str(output_file))
            for input_file, output_file in zip(input_files, output_files)
        ]
        return await asyncio.gather(*jobs, return_exceptions=True)

    n_success = 0
    n_total = 0
    results = asyncio.run(_process())
    for output_file, result in zip(output_files, results):
        n_total += 1
        if isinstance(result, Exception):
            click.echo(f"Error processing {output_file}: {result}", err=True)
        else:
            n_success += 1

    click.echo(
        f"Processed {n_total} files: {n_success} successes, {n_total - n_success} failures.",
        err=True,
    )


def main():
    cli()
